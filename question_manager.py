import json
import random
import os
from datetime import datetime

class QuestionManager:
    MASTERED_THRESHOLD = 3
    FOCUS_THRESHOLD = 3
    
    def __init__(self, questions_file):
        self.questions_file = questions_file
    
    def load_questions(self):
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_questions(self, questions):
        with open(self.questions_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
    
    def select_questions(self, all_questions, count, difficulty_manager):
        focus_questions = [q for q in all_questions if q.get('status') == 'focus']
        normal_questions = [q for q in all_questions if q.get('status', 'normal') == 'normal']
        
        weights = difficulty_manager.get_difficulty_weights()
        
        easy_qs = [q for q in normal_questions if q.get('difficulty') == '简单']
        medium_qs = [q for q in normal_questions if q.get('difficulty') == '中等']
        hard_qs = [q for q in normal_questions if q.get('difficulty') == '较难']
        
        selected = []
        used_ids = set()
        
        focus_count = min(len(focus_questions), int(count * 0.15))
        if focus_questions and focus_count > 0:
            random.shuffle(focus_questions)
            for q in focus_questions[:focus_count]:
                if q['id'] not in used_ids:
                    selected.append(q)
                    used_ids.add(q['id'])
        
        remaining = count - len(selected)
        
        easy_count = min(int(remaining * weights['简单']), len(easy_qs))
        medium_count = min(int(remaining * weights['中等']), len(medium_qs))
        hard_count = min(remaining - easy_count - medium_count, len(hard_qs))
        
        if easy_qs:
            random.shuffle(easy_qs)
            for q in easy_qs[:easy_count]:
                if q['id'] not in used_ids:
                    selected.append(q)
                    used_ids.add(q['id'])
        
        if medium_qs:
            random.shuffle(medium_qs)
            for q in medium_qs[:medium_count]:
                if q['id'] not in used_ids:
                    selected.append(q)
                    used_ids.add(q['id'])
        
        if hard_qs:
            random.shuffle(hard_qs)
            for q in hard_qs[:hard_count]:
                if q['id'] not in used_ids:
                    selected.append(q)
                    used_ids.add(q['id'])
        
        while len(selected) < count:
            remaining_qs = [q for q in all_questions if q['id'] not in used_ids]
            if not remaining_qs:
                break
            q = random.choice(remaining_qs)
            selected.append(q)
            used_ids.add(q['id'])
        
        random.shuffle(selected)
        
        for i, q in enumerate(selected):
            q['display_id'] = i + 1
        
        return selected
    
    def select_questions_by_distribution(self, all_questions, distribution, count):
        selected = []
        used_ids = set()
        
        focus_questions = [q for q in all_questions if q.get('status') == 'focus']
        random.shuffle(focus_questions)
        
        focus_count = min(len(focus_questions), int(count * 0.15))
        for q in focus_questions[:focus_count]:
            if q['id'] not in used_ids:
                selected.append(q)
                used_ids.add(q['id'])
        
        for category, cat_count in distribution.items():
            if cat_count < 1:
                continue
            
            cat_questions = [q for q in all_questions 
                           if q.get('category') == category and q['id'] not in used_ids]
            
            if not cat_questions:
                continue
            
            actual_count = min(int(cat_count), len(cat_questions))
            random.shuffle(cat_questions)
            
            for q in cat_questions[:actual_count]:
                selected.append(q)
                used_ids.add(q['id'])
        
        while len(selected) < count:
            remaining_qs = [q for q in all_questions if q['id'] not in used_ids]
            if not remaining_qs:
                break
            q = random.choice(remaining_qs)
            selected.append(q)
            used_ids.add(q['id'])
        
        random.shuffle(selected)
        
        for i, q in enumerate(selected):
            q['display_id'] = i + 1
        
        return selected[:count]
    
    def update_question_stats(self, question_id, is_correct):
        questions = self.load_questions()
        updated_question = None
        
        for q in questions:
            if q['id'] == question_id:
                if is_correct:
                    q['correct_count'] = q.get('correct_count', 0) + 1
                else:
                    q['wrong_count'] = q.get('wrong_count', 0) + 1
                
                q['last_answered'] = datetime.now().isoformat()
                
                if q.get('correct_count', 0) >= self.MASTERED_THRESHOLD:
                    q['status'] = 'mastered'
                elif q.get('wrong_count', 0) >= self.FOCUS_THRESHOLD:
                    q['status'] = 'focus'
                
                updated_question = q
                break
        
        self.save_questions(questions)
        return updated_question
    
    def remove_mastered_questions(self):
        questions = self.load_questions()
        mastered_ids = [q['id'] for q in questions if q.get('status') == 'mastered']
        questions = [q for q in questions if q.get('status') != 'mastered']
        self.save_questions(questions)
        return mastered_ids
    
    def add_new_questions(self, new_questions):
        questions = self.load_questions()
        max_id = max(q['id'] for q in questions) if questions else 0
        
        for i, q in enumerate(new_questions):
            q['id'] = max_id + i + 1
            q['correct_count'] = 0
            q['wrong_count'] = 0
            q['seen'] = 0
            q['status'] = 'normal'
            questions.append(q)
        
        self.save_questions(questions)
        return len(new_questions)
    
    def get_statistics(self):
        questions = self.load_questions()
        
        mastered = sum(1 for q in questions if q.get('status') == 'mastered')
        focus = sum(1 for q in questions if q.get('status') == 'focus')
        normal = sum(1 for q in questions if q.get('status', 'normal') == 'normal')
        
        categories = {}
        for q in questions:
            cat = q.get('category', '未知')
            categories[cat] = categories.get(cat, 0) + 1
        
        difficulties = {}
        for q in questions:
            diff = q.get('difficulty', '中等')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        return {
            'total': len(questions),
            'mastered': mastered,
            'focus': focus,
            'normal': normal,
            'categories': categories,
            'difficulties': difficulties
        }
