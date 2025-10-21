from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt
from pathlib import Path
import os
import uuid

# === CORS & APP SETUP ===
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
import os
import motor.motor_asyncio

# Allowed origins (frontend URLs)
origins = [
    "https://eduhub-frontend.vercel.app",
    "http://localhost:3000",  # For local development
]

# Initialize FastAPI app
app = FastAPI()

# Apply CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ENVIRONMENT CONFIG ===
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALGO = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

if not MONGO_URL:
    raise ValueError("❌ Missing MONGO_URL in .env file")

print(f"✅ Using MongoDB URI: {MONGO_URL}")

# === DATABASE CONNECTION ===
client = motor.motor_asyncio.AsyncIOMotorClient(
    MONGO_URL,
    tls=True,
    tlsAllowInvalidCertificates=True,  # Fix for Windows/Render
    serverSelectionTimeoutMS=20000,
)
db = client[DB_NAME]


# APP
app = FastAPI()
api = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Helpers
Role = Literal['student', 'teacher', 'admin']
Subjects = Literal['Math', 'Science', 'English', 'Social Studies']

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# Models
class Token(BaseModel):
    access_token: str
    token_type: str = 'bearer'

class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: Role
    points: int = 0
    badges: List[str] = []
    class_grade: Optional[int] = None

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Role = 'student'
    class_grade: Optional[int] = Field(default=None, ge=1, le=12)

class UserClassUpdate(BaseModel):
    class_grade: int = Field(ge=1, le=12)

class AuthUser(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: Role
    points: int = 0
    badges: List[str] = []
    class_grade: Optional[int] = None

class Announcement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    body: str
    created_at: str = Field(default_factory=now_iso)
    created_by: str
    class_grade: Optional[int] = Field(default=None, ge=1, le=12)
    subject: Optional[Subjects] = None

class AnnouncementCreate(BaseModel):
    title: str
    body: str
    class_grade: Optional[int] = Field(default=None, ge=1, le=12)
    subject: Optional[Subjects] = None

class Doubt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    body: str
    created_by: str
    created_at: str = Field(default_factory=now_iso)
    anonymous: bool = False
    status: Literal['open', 'resolved'] = 'open'
    answer: Optional[str] = None
    answered_by: Optional[str] = None
    answered_at: Optional[str] = None

class Question(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    options: List[str]
    answer_index: int
    topic: str

class Quiz(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    subject: Subjects
    class_grade: int = Field(ge=1, le=12)
    topic: str
    questions: List[Question]
    created_by: str
    created_at: str = Field(default_factory=now_iso)
    approved: bool = False

class Submission(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    quiz_id: str
    student_id: str
    score: float
    total: int
    correct_count: int
    incorrect_count: int
    topic_breakdown: Dict[str, Dict[str, int]]
    created_at: str = Field(default_factory=now_iso)

class ChatThread(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    participants: List[Dict[str, str]]
    created_at: str = Field(default_factory=now_iso)
    type: Literal['teacher-student', 'teacher-admin']

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str
    sender_id: str
    sender_role: Role
    body: str
    created_at: str = Field(default_factory=now_iso)

# Sanitizers

def safe_user(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc.get("id", str(uuid.uuid4())),
        "name": doc.get("name", "User"),
        "email": doc.get("email", "unknown@example.com"),
        "role": doc.get("role", "student"),
        "points": doc.get("points", 0) or 0,
        "badges": doc.get("badges", []) or [],
        "class_grade": doc.get("class_grade"),
    }

def safe_quiz(doc: Dict[str, Any]) -> Dict[str, Any]:
    qs = []
    for q in (doc.get("questions", []) or []):
        qs.append({
            "id": q.get("id", str(uuid.uuid4())),
            "prompt": q.get("prompt", "Question"),
            "options": q.get("options", ["A","B","C","D"]) or ["A","B","C","D"],
            "answer_index": int(q.get("answer_index", 0)),
            "topic": q.get("topic", doc.get("topic", "General")),
        })
    if not qs:
        qs = [{"id": str(uuid.uuid4()), "prompt": "Placeholder", "options": ["A","B","C","D"], "answer_index": 0, "topic": doc.get("topic", "General")}]
    return {
        "id": doc.get("id", str(uuid.uuid4())),
        "title": doc.get("title", "Untitled Quiz"),
        "subject": doc.get("subject", "Math"),
        "class_grade": int(doc.get("class_grade", 1) or 1),
        "topic": doc.get("topic", "General"),
        "questions": qs,
        "created_by": doc.get("created_by", ""),
        "created_at": doc.get("created_at", now_iso()),
        "approved": bool(doc.get("approved", False)),
    }

def safe_doubt(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc.get("id", str(uuid.uuid4())),
        "title": doc.get("title", "Doubt"),
        "body": doc.get("body", ""),
        "created_by": doc.get("created_by", ""),
        "created_at": doc.get("created_at", now_iso()),
        "anonymous": bool(doc.get("anonymous", False)),
        "status": doc.get("status", "open"),
        "answer": doc.get("answer"),
        "answered_by": doc.get("answered_by"),
        "answered_at": doc.get("answered_at"),
    }

def safe_announcement(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc.get("id", str(uuid.uuid4())),
        "title": doc.get("title", "Announcement"),
        "body": doc.get("body", ""),
        "created_at": doc.get("created_at", now_iso()),
        "created_by": doc.get("created_by", ""),
        "class_grade": doc.get("class_grade"),
        "subject": doc.get("subject"),
    }

# Auth helpers

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user_by_email(email: str) -> Optional[dict]:
    return await db.users.find_one({"email": email})

async def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGO)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> AuthUser:
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise credentials_exception
    return AuthUser(**safe_user(user))

def require_roles(*roles: Role):
    async def _role_dep(user: AuthUser = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return _role_dep

# Routes: Health
@api.get("/")
async def root():
    return {"message": "EduPulse v2 API running"}

# Auth
@api.post("/auth/register", response_model=UserOut)
async def register(body: UserCreate):
    existing = await get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    doc = {
        "id": str(uuid.uuid4()),
        "name": body.name,
        "email": body.email,
        "role": body.role,
        "password_hash": get_password_hash(body.password),
        "created_at": now_iso(),
        "points": 0,
        "badges": [],
        "class_grade": body.class_grade if body.role == 'student' else None,
    }
    await db.users.insert_one(doc)
    return UserOut(**{k: doc.get(k) for k in ["id","name","email","role","points","badges","class_grade"]})

@api.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user.get("password_hash","")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    token = await create_access_token({"sub": user["id"], "role": user["role"]})
    return Token(access_token=token)

@api.get("/auth/me", response_model=UserOut)
async def me(user: AuthUser = Depends(get_current_user)):
    return UserOut(**user.model_dump())

@api.patch("/users/me", response_model=UserOut)
async def update_me_class(body: UserClassUpdate, user: AuthUser = Depends(require_roles('student'))):
    await db.users.update_one({"id": user.id}, {"$set": {"class_grade": body.class_grade}})
    u = await db.users.find_one({"id": user.id}, {"_id":0})
    return UserOut(**safe_user(u))

# Announcements (admin only)
@api.get("/announcements", response_model=List[Announcement])
async def list_announcements(user: AuthUser = Depends(get_current_user)):
    q: Dict[str, Any] = {}
    if user.role == 'student' and user.class_grade:
        q = {"$or": [{"class_grade": None}, {"class_grade": user.class_grade}]}
    docs = await db.announcements.find(q, {"_id": 0}).sort("created_at", -1).to_list(200)
    return [Announcement(**safe_announcement(d)) for d in docs]

@api.post("/announcements", response_model=Announcement)
async def create_announcement(body: AnnouncementCreate, user: AuthUser = Depends(require_roles('admin'))):
    doc = Announcement(title=body.title, body=body.body, class_grade=body.class_grade, subject=body.subject, created_by=user.id)
    await db.announcements.insert_one(doc.model_dump())
    return doc

# Doubts
class DoubtCreate(BaseModel):
    title: str
    body: str
    anonymous: bool = False

@api.get("/doubts", response_model=List[Doubt])
async def list_doubts(page: int = Query(default=1, ge=1), page_size: int = Query(default=10, ge=1, le=50), _: AuthUser = Depends(get_current_user)):
    skip = (page - 1) * page_size
    cursor = db.doubts.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(page_size)
    docs = await cursor.to_list(length=page_size)
    return [Doubt(**safe_doubt(d)) for d in docs]

@api.post("/doubts", response_model=Doubt)
async def create_doubt(body: DoubtCreate, user: AuthUser = Depends(require_roles('student','teacher','admin'))):
    d = Doubt(title=body.title, body=body.body, anonymous=body.anonymous, created_by=user.id)
    await db.doubts.insert_one(d.model_dump())
    return d

@api.post("/doubts/{doubt_id}/answer", response_model=Doubt)
async def answer_doubt(doubt_id: str, body: DoubtCreate, user: AuthUser = Depends(require_roles('teacher','admin'))):
    existing = await db.doubts.find_one({"id": doubt_id}, {"_id": 0})
    if not existing:
        raise HTTPException(404, "Doubt not found")
    existing.update({
        "answer": body.body,
        "answered_by": user.id,
        "answered_at": now_iso(),
        "status": "resolved"
    })
    await db.doubts.update_one({"id": doubt_id}, {"$set": existing})
    return Doubt(**safe_doubt(existing))

@api.delete("/doubts/{doubt_id}")
async def delete_doubt(doubt_id: str, _: AuthUser = Depends(require_roles('admin'))):
    res = await db.doubts.delete_one({"id": doubt_id})
    if res.deleted_count == 0:
        raise HTTPException(404, "Doubt not found")
    return {"status": "deleted"}

# Quizzes
class QuizCreate(BaseModel):
    title: str
    subject: Subjects
    class_grade: int = Field(ge=1, le=12)
    topic: str
    questions: List[Question]

@api.get("/quizzes", response_model=List[Quiz])
async def list_quizzes(
    user: AuthUser = Depends(get_current_user),
    subject: Optional[Subjects] = None,
    class_grade: Optional[int] = Query(default=None, ge=1, le=12),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
):
    q: Dict[str, Any] = {}
    if subject:
        q["subject"] = subject
    if class_grade is not None:
        q["class_grade"] = class_grade
    if user.role == 'student':
        q["approved"] = True
    skip = (page - 1) * page_size
    cursor = db.quizzes.find(q, {"_id": 0}).sort("created_at", -1).skip(skip).limit(page_size)
    docs = await cursor.to_list(length=page_size)
    return [Quiz(**safe_quiz(d)) for d in docs]

@api.post("/quizzes", response_model=Quiz)
async def create_quiz(body: QuizCreate, user: AuthUser = Depends(require_roles('teacher','admin'))):
    if len(body.questions) < 10:
        raise HTTPException(400, "Quiz must have at least 10 questions")
    approved = True if user.role == 'admin' else False
    qz = Quiz(title=body.title, subject=body.subject, class_grade=body.class_grade, topic=body.topic, questions=body.questions, created_by=user.id, approved=approved)
    await db.quizzes.insert_one(qz.model_dump())
    return qz

@api.get("/quizzes/{quiz_id}", response_model=Quiz)
async def get_quiz(quiz_id: str, user: AuthUser = Depends(get_current_user)):
    doc = await db.quizzes.find_one({"id": quiz_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Quiz not found")
    sq = safe_quiz(doc)
    if user.role == 'student' and not sq.get('approved', False):
        raise HTTPException(403, "Quiz pending approval")
    return Quiz(**sq)

@api.delete("/quizzes/{quiz_id}")
async def delete_quiz(quiz_id: str, _: AuthUser = Depends(require_roles('admin'))):
    res = await db.quizzes.delete_one({"id": quiz_id})
    if res.deleted_count == 0:
        raise HTTPException(404, "Quiz not found")
    await db.submissions.delete_many({"quiz_id": quiz_id})
    return {"status": "deleted"}

@api.post("/quizzes/{quiz_id}/approve")
async def approve_quiz(quiz_id: str, _: AuthUser = Depends(require_roles('admin'))):
    res = await db.quizzes.update_one({"id": quiz_id}, {"$set": {"approved": True}})
    if res.matched_count == 0:
        raise HTTPException(404, "Quiz not found")
    return {"status": "approved"}

class QuizSubmit(BaseModel):
    answers: Dict[str, int]

class QuizFeedback(BaseModel):
    submission: Submission
    feedback: Dict[str, Any]
    new_points: int
    badges: List[str]

@api.post("/quizzes/{quiz_id}/submit", response_model=QuizFeedback)
async def submit_quiz(quiz_id: str, body: QuizSubmit, user: AuthUser = Depends(require_roles('student'))):
    quiz_doc = await db.quizzes.find_one({"id": quiz_id}, {"_id": 0})
    if not quiz_doc:
        raise HTTPException(404, "Quiz not found")
    sq = safe_quiz(quiz_doc)
    if not sq.get('approved', False):
        raise HTTPException(403, "Quiz pending approval")
    quiz = Quiz(**sq)
    correct = 0
    incorrect = 0
    topic_breakdown: Dict[str, Dict[str, int]] = {}
    for q in quiz.questions:
        chosen = body.answers.get(q.id, -1)
        is_correct = int(chosen == q.answer_index)
        correct += is_correct
        incorrect += (1 - is_correct)
        topic_breakdown.setdefault(q.topic, {"correct": 0, "incorrect": 0})
        if is_correct:
            topic_breakdown[q.topic]["correct"] += 1
        else:
            topic_breakdown[q.topic]["incorrect"] += 1
    score = round((correct / len(quiz.questions)) * 100, 2)
    sub = Submission(quiz_id=quiz.id, student_id=user.id, score=score, total=len(quiz.questions), correct_count=correct, incorrect_count=incorrect, topic_breakdown=topic_breakdown)
    await db.submissions.insert_one(sub.model_dump())
    # Points & badges
    inc = 10 + int(score // 20)
    await db.users.update_one({"id": user.id}, {"$inc": {"points": inc}})
    udoc = await db.users.find_one({"id": user.id}, {"_id": 0})
    badges = set((udoc or {}).get('badges', []) or [])
    total_subs = await db.submissions.count_documents({"student_id": user.id})
    if total_subs >= 1:
        badges.add('Getting Started')
    if score >= 90:
        badges.add('Ace')
    if (udoc or {}).get('points', 0) >= 100:
        badges.add('Century Club')
    await db.users.update_one({"id": user.id}, {"$set": {"badges": list(badges)}})
    weak_topics = [t for t, v in topic_breakdown.items() if v.get('incorrect',0) > v.get('correct',0)]
    more_quizzes = await db.quizzes.find({"class_grade": quiz.class_grade, "subject": quiz.subject, "approved": True}, {"_id": 0, "id":1, "title":1, "topic":1}).to_list(50)
    recs = [{"topic": t, "suggested_quizzes": [q for q in more_quizzes if q['topic']==t][:2], "tips": ["Watch recap", "Do spaced repetition", "Practice 10 Qs"]} for t in weak_topics]
    feedback = {"weak_topics": weak_topics, "recommendations": recs}
    return QuizFeedback(submission=sub, feedback=feedback, new_points=inc, badges=list(badges))

# Dashboards
class StudentDashboard(BaseModel):
    recent_score: Optional[float] = None
    attempted_quizzes: int
    weak_topics: List[str] = []
    points: int = 0
    badges: List[str] = []

@api.get("/dashboard/student", response_model=StudentDashboard)
async def dash_student(user: AuthUser = Depends(require_roles('student'))):
    subs = await db.submissions.find({"student_id": user.id}, {"_id": 0}).sort("created_at", -1).to_list(50)
    recent_score = subs[0]["score"] if subs else None
    weak: Dict[str, int] = {}
    for s in subs:
        for t, v in s.get('topic_breakdown', {}).items():
            if v.get('incorrect',0) > v.get('correct',0):
                weak[t] = weak.get(t,0) + 1
    udoc = await db.users.find_one({"id": user.id}, {"_id": 0})
    u = safe_user(udoc or {})
    return StudentDashboard(recent_score=recent_score, attempted_quizzes=len(subs), weak_topics=list(weak.keys()), points=u['points'], badges=u['badges'])

class TeacherDashboard(BaseModel):
    quizzes_created: int
    pending_doubts: int
    class_performance: Dict[str, float]

@api.get("/dashboard/teacher", response_model=TeacherDashboard)
async def dash_teacher(user: AuthUser = Depends(require_roles('teacher','admin'))):
    quizzes = await db.quizzes.find({"created_by": user.id}, {"_id": 0, "id":1, "class_grade":1}).to_list(500)
    qids = [q.get('id') for q in quizzes if q.get('id')]
    perf: Dict[str, List[float]] = {}
    if qids:
        subs = await db.submissions.find({"quiz_id": {"$in": qids}}, {"_id": 0, "score":1, "quiz_id":1}).to_list(2000)
        class_map = {q.get('id'): (q.get('class_grade') or 0) for q in quizzes}
        for s in subs:
            cg = class_map.get(s.get('quiz_id'))
            if cg is not None:
                perf.setdefault(str(cg), []).append(s.get('score',0.0))
    class_avg = {k: round(sum(v)/len(v),2) if v else 0.0 for k,v in perf.items()}
    pending = await db.doubts.count_documents({"status": "open"})
    return TeacherDashboard(quizzes_created=len(quizzes), pending_doubts=pending, class_performance=class_avg)

class AdminDashboard(BaseModel):
    total_quizzes: int
    total_announcements: int
    chat_threads: int
    messages: int

@api.get("/dashboard/admin", response_model=AdminDashboard)
async def dash_admin(_: AuthUser = Depends(require_roles('admin'))):
    tq = await db.quizzes.count_documents({})
    ta = await db.announcements.count_documents({})
    th = await db.chat_threads.count_documents({})
    ms = await db.chat_messages.count_documents({})
    return AdminDashboard(total_quizzes=tq, total_announcements=ta, chat_threads=th, messages=ms)

@api.post("/admin/clear-stats")
async def admin_clear_stats(_: AuthUser = Depends(require_roles('admin'))):
    await db.submissions.delete_many({})
    await db.users.update_many({}, {"$set": {"points": 0, "badges": []}})
    return {"status": "cleared"}

class StudentPerf(BaseModel):
    student_id: str
    name: str
    email: EmailStr
    attempts: int
    avg_score: float
    weak_topics: List[str] = []

@api.get("/teacher/students", response_model=List[StudentPerf])
async def teacher_students(class_grade: Optional[int] = Query(default=None, ge=1, le=12), user: AuthUser = Depends(require_roles('teacher','admin'))):
    quizzes = await db.quizzes.find({"created_by": user.id}, {"_id":0, "id":1}).to_list(1000)
    qids = [q.get('id') for q in quizzes if q.get('id')]
    query = {"quiz_id": {"$in": qids}} if qids else {"quiz_id": "__none__"}
    subs = await db.submissions.find(query, {"_id":0}).to_list(5000)
    perf_map: Dict[str, Dict[str, Any]] = {}
    for s in subs:
        sid = s.get('student_id')
        d = perf_map.setdefault(sid, {"scores": [], "weak": {}})
        d["scores"].append(s.get('score', 0.0))
        for t, v in (s.get('topic_breakdown', {}) or {}).items():
            if v.get('incorrect',0) > v.get('correct',0):
                d["weak"][t] = d["weak"].get(t,0) + 1
    result: List[StudentPerf] = []
    for sid, stats in perf_map.items():
        stu = await db.users.find_one({"id": sid}, {"_id":0})
        if not stu:
            continue
        if class_grade and stu.get('class_grade') != class_grade:
            continue
        attempts = len(stats['scores'])
        avg = round(sum(stats['scores'])/attempts, 2) if attempts else 0.0
        weak_topics = sorted(stats['weak'], key=stats['weak'].get, reverse=True)[:3]
        result.append(StudentPerf(student_id=sid, name=stu.get('name',''), email=stu.get('email',''), attempts=attempts, avg_score=avg, weak_topics=weak_topics))
    return result

# Chat
class ThreadCreate(BaseModel):
    target_email: EmailStr

@api.get("/chat/threads", response_model=List[ChatThread])
async def list_threads(user: AuthUser = Depends(get_current_user)):
    docs = await db.chat_threads.find({"participants.user_id": user.id}, {"_id": 0}).sort("created_at", -1).to_list(200)
    return [ChatThread(**d) for d in docs]

@api.post("/chat/threads", response_model=ChatThread)
async def create_thread(body: ThreadCreate, user: AuthUser = Depends(get_current_user)):
    target = await get_user_by_email(body.target_email)
    if not target:
        raise HTTPException(404, "Target user not found")
    if user.role == 'student' and target['role'] != 'teacher':
        raise HTTPException(403, "Students can only chat with teachers")
    if user.role == 'admin' and target['role'] != 'teacher':
        raise HTTPException(403, "Admins can only chat with teachers")
    if user.role == 'teacher' and target['role'] not in ['student','admin']:
        raise HTTPException(403, "Teachers can chat with students or admins only")
    chat_type = 'teacher-student' if 'student' in [user.role, target['role']] else 'teacher-admin'
    pset = sorted([user.id, target['id']])
    existing = await db.chat_threads.find_one({"participants": {"$all": [
        {"$elemMatch": {"user_id": pset[0]}}, {"$elemMatch": {"user_id": pset[1]}}
    ]}}, {"_id":0})
    if existing:
        return ChatThread(**existing)
    thread = ChatThread(participants=[{"user_id": user.id, "role": user.role},{"user_id": target['id'], "role": target['role']}], type=chat_type)
    await db.chat_threads.insert_one(thread.model_dump())
    return thread

class MessageCreate(BaseModel):
    thread_id: str
    body: str

@api.get("/chat/messages", response_model=List[ChatMessage])
async def get_messages(thread_id: str, user: AuthUser = Depends(get_current_user)):
    thr = await db.chat_threads.find_one({"id": thread_id}, {"_id": 0})
    if not thr or all(p.get('user_id') != user.id for p in thr.get('participants', [])):
        raise HTTPException(404, "Thread not found")
    docs = await db.chat_messages.find({"thread_id": thread_id}, {"_id": 0}).sort("created_at", 1).to_list(1000)
    return [ChatMessage(**d) for d in docs]

@api.post("/chat/messages", response_model=ChatMessage)
async def post_message(body: MessageCreate, user: AuthUser = Depends(get_current_user)):
    thr = await db.chat_threads.find_one({"id": body.thread_id}, {"_id": 0})
    if not thr or all(p.get('user_id') != user.id for p in thr.get('participants', [])):
        raise HTTPException(404, "Thread not found")
    msg = ChatMessage(thread_id=body.thread_id, sender_id=user.id, sender_role=user.role, body=body.body)
    await db.chat_messages.insert_one(msg.model_dump())
    return msg

# Student quiz statistics
class QuizAttempt(BaseModel):
    quiz_id: str
    title: str
    score: float
    created_at: str

@api.get("/stats/quizzes", response_model=List[QuizAttempt])
async def stats_quizzes(user: AuthUser = Depends(require_roles('student'))):
    subs = await db.submissions.find({"student_id": user.id}, {"_id":0}).sort("created_at", -1).to_list(200)
    attempts: List[QuizAttempt] = []
    for s in subs:
        q = await db.quizzes.find_one({"id": s.get('quiz_id')}, {"_id":0, "title":1})
        attempts.append(QuizAttempt(quiz_id=s.get('quiz_id',''), title=(q or {}).get('title','Quiz'), score=s.get('score',0.0), created_at=s.get('created_at', now_iso())))
    return attempts

# Seed data
async def seed():
    if await db.users.count_documents({}) == 0:
        admin = {"id": str(uuid.uuid4()), "name": "Admin", "email": "admin@school.com", "role": "admin", "password_hash": get_password_hash("admin123"), "created_at": now_iso(), "points": 0, "badges": []}
        teacher = {"id": str(uuid.uuid4()), "name": "Ms. Smith", "email": "teacher@school.com", "role": "teacher", "password_hash": get_password_hash("teacher123"), "created_at": now_iso(), "points": 0, "badges": []}
        student = {"id": str(uuid.uuid4()), "name": "John Doe", "email": "student@school.com", "role": "student", "password_hash": get_password_hash("student123"), "created_at": now_iso(), "points": 0, "badges": [], "class_grade": 7}
        await db.users.insert_many([admin, teacher, student])
        await db.announcements.insert_one(Announcement(title="Welcome to EduPulse v2", body="Explore your dashboards and chats.", created_by=admin['id']).model_dump())
    if await db.quizzes.count_documents({}) == 0:
        u_teacher = await db.users.find_one({"role": "teacher"}, {"_id":0})
        subjects = ['Math','Science','English','Social Studies']
        quizzes: List[Dict[str, Any]] = []
        for grade in range(1,13):
            for subj in subjects:
                topic = f"Core {subj} G{grade}"
                questions: List[Question] = []
                for i in range(10):
                    if subj == 'Math':
                        prompt = f"G{grade}: What is {i+1} + {i+2}?"
                        options = [str(i), str(i+2), str((i+1)+(i+2)), str(i+5)]
                        ans = 2
                        tpc = 'Arithmetic'
                    elif subj == 'Science':
                        prompt = f"G{grade}: Which is a planet?"
                        options = ["Mars","Tree","Rock","River"]
                        ans = 0
                        tpc = 'Astronomy'
                    elif subj == 'English':
                        prompt = f"G{grade}: Choose the noun"
                        options = ["Quickly","Happiness","Blue","Run"]
                        ans = 1
                        tpc = 'Grammar'
                    else:
                        prompt = f"G{grade}: Capital of India?"
                        options = ["Mumbai","Delhi","Kolkata","Chennai"]
                        ans = 1
                        tpc = 'Geography'
                    questions.append(Question(prompt=prompt, options=options, answer_index=ans, topic=tpc))
                qz = Quiz(title=f"{subj} Quiz G{grade}", subject=subj, class_grade=grade, topic=topic, questions=questions, created_by=u_teacher['id'], approved=True)
                quizzes.append(qz.model_dump())
        if quizzes:
            await db.quizzes.insert_many(quizzes)

@app.on_event("startup")
async def on_start():
    await seed()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# Mount router
app.include_router(api)
