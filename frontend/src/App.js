import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";
import "@/App.css";
import { Analytics } from '@vercel/analytics/react';
// ------------------------ CONFIG ------------------------
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
axios.defaults.baseURL = `${BACKEND_URL}/api`;

// Axios global error handling
axios.interceptors.response.use(
  (response) => response,
  (err) => {
    const msg = err?.response?.data?.detail || err?.response?.data || err.message || "Something went wrong";
    toast.error(msg);
    return Promise.reject(err);
  }
);


// Removed Python code as it does not belong in a JavaScript file.


// ------------------------ AUTH ------------------------
export function setAuth(token) {
  if (token) {
    axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
    localStorage.setItem("token", token);
  } else {
    delete axios.defaults.headers.common["Authorization"];
    localStorage.removeItem("token");
  }
}

export function useAuth() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      setAuth(token);
      axios.get("/auth/me").then((res) => setUser(res.data)).catch(() => setAuth(null));
    }
  }, []);

  return { user, setUser };
}

// ------------------------ THEME TOGGLE ------------------------
function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  return (
    <div className="flex items-center gap-2">
      <Switch checked={dark} onCheckedChange={setDark} />
      <span className="text-sm">{dark ? "Dark" : "Light"}</span>
    </div>
  );
}

// ------------------------ NAVBAR ------------------------
function Nav({ user, onLogout }) {
  return (
    <div className="sticky top-0 z-20 backdrop-blur bg-background/70 border-b">
      <div className="max-w-6xl mx-auto flex items-center justify-between py-3 px-4">
        <Link to="/" className="font-semibold tracking-tight">EduPulse</Link>
        <div className="flex items-center gap-4">
          <Link to="/announcements" className="text-sm">Announcements</Link>
          <Link to="/doubts" className="text-sm">Doubt Board</Link>
          <Link to="/quizzes" className="text-sm">Quizzes</Link>
          {user?.role === "student" && <Link to="/quizzes-stats" className="text-sm">Stats</Link>}
          <Link to="/chat" className="text-sm">Chat</Link>
          <ThemeToggle />
          {user && (
            <div className="flex items-center gap-2">
              <span className="text-sm">{user.name} · {user.role}</span>
              <Button variant="secondary" onClick={onLogout}>Logout</Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ------------------------ CONSTANTS ------------------------
const SUBJECTS = ["Math", "Science", "English", "Social Studies"];
const CLASSES = Array.from({ length: 12 }, (_, i) => i + 1);

// ------------------------ LOGIN COMPONENT ------------------------
function Login({ setUser }) {
  const [email, setEmail] = useState("student@school.com");
  const [password, setPassword] = useState("student123");
  const [classPick, setClassPick] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    try {
      const fd = new URLSearchParams();
      fd.append("username", email);
      fd.append("password", password);

      const res = await axios.post("/auth/login", fd, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
      });

      setAuth(res.data.access_token);
      let me = (await axios.get("/auth/me")).data;

      if (me.role === "student" && classPick) {
        await axios.patch("/users/me", { class_grade: Number(classPick) });
        me = (await axios.get("/auth/me")).data;
      }

      setUser(me);
      toast.success("Logged in");
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Card className="max-w-md mx-auto mt-10">
      <CardHeader><CardTitle>Welcome to EduPulse</CardTitle></CardHeader>
      <CardContent>
        <form onSubmit={submit} className="space-y-4">
          <div>
            <Label htmlFor="email">Email</Label>
            <Input id="email" value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <Label htmlFor="password">Password</Label>
            <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          <div className="grid grid-cols-2 gap-2 items-end">
            <div>
              <Label>Student Class (optional)</Label>
              <Select value={classPick} onValueChange={setClassPick}>
                <SelectTrigger><SelectValue placeholder="Select class" /></SelectTrigger>
                <SelectContent>
                  {CLASSES.map((c) => <SelectItem key={c} value={String(c)}>Class {c}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="text-xs text-muted-foreground">If student, select class to personalize</div>
          </div>
          <Button className="w-full" type="submit">Login</Button>
        </form>
        <div className="text-xs text-muted-foreground mt-4">
          Seed accounts: student@school.com/student123 · teacher@school.com/teacher123 · admin@school.com/admin123
        </div>
      </CardContent>
    </Card>
  );
}

// ------------------------ DASHBOARDS (PLACEHOLDER EXAMPLES) ------------------------
function Home({ user }) {
  if (!user) return <Navigate to="/login" replace />;
  if (user.role === "student") return <div>Student Dashboard</div>;
  if (user.role === "admin") return <div>Admin Dashboard</div>;
  return <div>Teacher Dashboard</div>;
}

// ------------------------ MAIN APP ------------------------
function App() {
  const { user, setUser } = useAuth();

  const logout = () => {
    setAuth(null);
    setUser(null);
  };

  useEffect(() => {
    if (user) toast.success(`Hello, ${user.name}`);
  }, [user]);

  return (
    <div className="App min-h-screen">
      <BrowserRouter>
        <Nav user={user} onLogout={logout} />
        <Routes>
          <Route path="/" element={<Home user={user} />} />
          <Route path="/login" element={user ? <Navigate to="/" replace /> : <Login setUser={setUser} />} />
          {/* Add your other routes here */}
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" />
    </div>
  );
}

export default App;
