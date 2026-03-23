import json
import os
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import threading

class SessionManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.sessions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions.json')
        self._load_sessions()
    
    def _load_sessions(self):
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    self.sessions = json.load(f)
            except:
                self.sessions = {}
    
    def _save_sessions(self):
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(self.sessions, f, ensure_ascii=False, indent=2)
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'question_ids': [],
            'answers': {},
            'current_index': 0,
            'category_weights': {},
            'answer_history': []
        }
        self._save_sessions()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        if session_id in self.sessions:
            self.sessions[session_id].update(data)
            self._save_sessions()
    
    def set_question_ids(self, session_id: str, question_ids: list):
        if session_id in self.sessions:
            self.sessions[session_id]['question_ids'] = question_ids
            self.sessions[session_id]['answers'] = {}
            self.sessions[session_id]['current_index'] = 0
            self._save_sessions()
    
    def get_question_ids(self, session_id: str) -> list:
        if session_id in self.sessions:
            return self.sessions[session_id].get('question_ids', [])
        return []
    
    def save_answer(self, session_id: str, question_id: int, answer: str):
        if session_id in self.sessions:
            self.sessions[session_id]['answers'][str(question_id)] = answer
            self._save_sessions()
    
    def get_answers(self, session_id: str) -> Dict[str, str]:
        if session_id in self.sessions:
            return self.sessions[session_id].get('answers', {})
        return {}
    
    def set_current_index(self, session_id: str, index: int):
        if session_id in self.sessions:
            self.sessions[session_id]['current_index'] = index
            self._save_sessions()
    
    def get_current_index(self, session_id: str) -> int:
        if session_id in self.sessions:
            return self.sessions[session_id].get('current_index', 0)
        return 0
    
    def add_answer_history(self, session_id: str, record: Dict[str, Any]):
        if session_id in self.sessions:
            self.sessions[session_id]['answer_history'].append(record)
            self._save_sessions()
    
    def get_answer_history(self, session_id: str) -> list:
        if session_id in self.sessions:
            return self.sessions[session_id].get('answer_history', [])
        return []
    
    def set_category_weights(self, session_id: str, weights: Dict[str, float]):
        if session_id in self.sessions:
            self.sessions[session_id]['category_weights'] = weights
            self._save_sessions()
    
    def get_category_weights(self, session_id: str) -> Dict[str, float]:
        if session_id in self.sessions:
            return self.sessions[session_id].get('category_weights', {})
        return {}
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        now = datetime.now()
        to_delete = []
        for sid, data in self.sessions.items():
            created = datetime.fromisoformat(data.get('created_at', now.isoformat()))
            age = (now - created).total_seconds() / 3600
            if age > max_age_hours:
                to_delete.append(sid)
        
        for sid in to_delete:
            del self.sessions[sid]
        
        if to_delete:
            self._save_sessions()
