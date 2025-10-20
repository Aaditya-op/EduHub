# EduPulse v2 — AI-Powered Learning Gap & Performance Tracker

FastAPI + React + MongoDB app for schools (classes 1–12). Identify gaps, run quizzes, manage doubts, announcements, and chats.

## What’s New in v2
- Role-based quiz workflow: teachers/admins create quizzes, admins approve; students see only approved quizzes
- Cleaner quiz creation UI and search/filter + pagination
- Students can set their class at login (personalized content)
- Teacher dashboard: per-student performance table (attempts, avg score, weak topics)
- Admin: delete doubts, approve/delete quizzes, class-level announcements
- Chat: teacher↔student/admin with timestamps; persisted in MongoDB
- Badges/points on quiz completion; improved dark theme legibility

## Core Features
- Auth (JWT) with roles: student, teacher, admin (bcrypt hashing)
- Quizzes: 12×4 seed quizzes (10+ Q each), create/attempt/approve/delete
- Doubt Board: post, answer, resolve; admin delete
- Announcements: global or targeted by class/subject (admin only)
- Dashboards: student, teacher (class analytics + students’ table), admin overview
- Recommendations: simulated “weak topics” and suggested quizzes

## Tech Stack
- Backend: FastAPI, Motor (MongoDB), python-jose (JWT), passlib[bcrypt]
- Frontend: React, Tailwind, shadcn/ui, sonner

## Environment
- backend/.env: MONGO_URL, DB_NAME, JWT_SECRET, CORS_ORIGINS
- frontend/.env: REACT_APP_BACKEND_URL
Top-level `.env.example` provided.

## Local Setup
1) MongoDB running; update MONGO_URL if needed
2) Backend: cd app/backend; python -m venv .venv && source .venv/bin/activate; pip install -r requirements.txt; uvicorn server:app --host 0.0.0.0 --port 8001 --reload
3) Frontend: cd app/frontend; yarn; yarn start
Open http://localhost:3000

## Seed Users
- Admin: admin@school.com / admin123
- Teacher: teacher@school.com / teacher123
- Student: student@school.com / student123

## API Highlights (all prefixed with /api)
- Auth: POST /auth/login (OAuth2), GET /auth/me, PATCH /users/me (student class update)
- Quizzes: GET/POST/DELETE /quizzes, POST /quizzes/{id}/approve (admin), POST /quizzes/{id}/submit (student)
- Doubts: GET/POST /doubts, POST /doubts/{id}/answer (teacher/admin), DELETE /doubts/{id} (admin)
- Dashboards: GET /dashboard/student, /dashboard/teacher, /dashboard/admin, GET /teacher/students
- Chat: GET/POST /chat/threads, GET/POST /chat/messages
- Recommendations: GET /recommendations, Stats: GET /stats/quizzes

## Manual Testing Tips
- Login as teacher → Create quiz (10+ Q) → Login as admin to approve → Login as student to attempt
- Post doubt as student, answer as teacher, delete as admin
- Teacher dashboard → filter by class and check student table

## Future Ideas
- Export to CSV/PDF; question banks and randomization; timed quizzes; rubric feedback; multi-language UI
