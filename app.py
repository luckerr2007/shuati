import json
import os
import random
import re
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from difficulty_manager import DifficultyManager
from question_manager import QuestionManager
from dynamic_weight_manager import DynamicWeightManager
from random_forest_selector import RandomForestQuestionSelector
from local_optimizer import LocalOptimizer
from session_manager import SessionManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(BASE_DIR, 'questions.json')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
CATEGORY_STATS_FILE = os.path.join(BASE_DIR, 'category_stats.json')

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
app.secret_key = 'exam-system-secret-key-2026'
CORS(app, supports_credentials=True)

difficulty_manager = DifficultyManager()
question_manager = QuestionManager(QUESTIONS_FILE)
weight_manager = DynamicWeightManager(CATEGORY_STATS_FILE)
rf_selector = RandomForestQuestionSelector(QUESTIONS_FILE)
local_optimizer = LocalOptimizer()
session_manager = SessionManager()

def is_mobile_device(user_agent):
    mobile_patterns = [
        'Android', 'iPhone', 'iPad', 'iPod', 'Windows Phone', 
        'BlackBerry', 'Nokia', 'Opera Mini', 'Mobile', 'webOS'
    ]
    user_agent = user_agent.lower() if user_agent else ''
    for pattern in mobile_patterns:
        if pattern.lower() in user_agent:
            return True
    return False

@app.route('/')
def index():
    user_agent = request.headers.get('User-Agent', '')
    if is_mobile_device(user_agent):
        return send_from_directory('static', 'mobile.html')
    return send_from_directory('static', 'index.html')

@app.route('/mobile')
def mobile():
    return send_from_directory('static', 'mobile.html')

@app.route('/desktop')
def desktop():
    return send_from_directory('static', 'index.html')

@app.route('/api/session/create', methods=['POST'])
def create_session():
    session_id = session_manager.create_session()
    return jsonify({
        'success': True,
        'session_id': session_id
    })

@app.route('/api/session/restore', methods=['POST'])
def restore_session():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'success': False, 'error': '缺少session_id'})
    
    session_data = session_manager.get_session(session_id)
    if not session_data:
        return jsonify({'success': False, 'error': '会话不存在'})
    
    question_ids = session_manager.get_question_ids(session_id)
    answers = session_manager.get_answers(session_id)
    current_index = session_manager.get_current_index(session_id)
    
    all_questions = question_manager.load_questions()
    questions = [q for q in all_questions if q['id'] in question_ids]
    questions.sort(key=lambda x: question_ids.index(x['id']) if x['id'] in question_ids else 0)
    
    return jsonify({
        'success': True,
        'questions': questions,
        'answers': answers,
        'current_index': current_index,
        'total': len(questions)
    })

@app.route('/api/session/save', methods=['POST'])
def save_session():
    data = request.json
    session_id = data.get('session_id')
    question_ids = data.get('question_ids', [])
    answers = data.get('answers', {})
    current_index = data.get('current_index', 0)
    
    if not session_id:
        return jsonify({'success': False, 'error': '缺少session_id'})
    
    session_manager.set_question_ids(session_id, question_ids)
    for qid, ans in answers.items():
        session_manager.save_answer(session_id, int(qid), ans)
    session_manager.set_current_index(session_id, current_index)
    
    return jsonify({'success': True})

@app.route('/api/questions', methods=['GET'])
def get_questions():
    count = request.args.get('count', 100, type=int)
    session_id = request.args.get('session_id')
    
    existing_ids = []
    if session_id:
        existing_ids = session_manager.get_question_ids(session_id)
    
    if existing_ids:
        all_questions = question_manager.load_questions()
        questions = [q for q in all_questions if q['id'] in existing_ids]
        questions.sort(key=lambda x: existing_ids.index(x['id']) if x['id'] in existing_ids else 0)
        
        return jsonify({
            'success': True,
            'questions': questions,
            'total': len(questions),
            'restored': True
        })
    
    selected = rf_selector.select_questions(count, session_id)
    
    if session_id:
        question_ids = [q['id'] for q in selected]
        session_manager.set_question_ids(session_id, question_ids)
        
        weights = rf_selector.get_category_weights()
        session_manager.set_category_weights(session_id, weights)
    
    return jsonify({
        'success': True,
        'questions': selected,
        'total': len(selected),
        'restored': False
    })

@app.route('/api/submit', methods=['POST'])
def submit_answer():
    data = request.json
    question_id = data.get('question_id')
    user_answer = data.get('answer')
    session_id = data.get('session_id')
    
    all_questions = question_manager.load_questions()
    
    question = next((q for q in all_questions if q['id'] == question_id), None)
    
    if not question:
        return jsonify({'success': False, 'error': '题目不存在'}), 404
    
    is_correct = user_answer == question['answer']
    category = question.get('category', '未知')
    
    if is_correct:
        difficulty_manager.record_correct()
    else:
        difficulty_manager.record_wrong()
    
    weight_manager.record_answer(category, is_correct)
    
    rf_selector.update_weights_from_result(question_id, is_correct)
    
    if session_id:
        session_manager.save_answer(session_id, question_id, user_answer)
        
        session_manager.add_answer_history(session_id, {
            'question_id': question_id,
            'category': category,
            'is_correct': is_correct,
            'user_answer': user_answer,
            'difficulty': question.get('difficulty', '中等')
        })
    
    updated_question = question_manager.update_question_stats(question_id, is_correct)
    
    return jsonify({
        'success': True,
        'is_correct': is_correct,
        'correct_answer': question['answer'],
        'difficulty_level': difficulty_manager.get_current_level(),
        'category_difficulty': weight_manager.get_difficulty_for_category(category)
    })

@app.route('/api/results', methods=['POST'])
def get_results():
    data = request.json
    answers = data.get('answers', [])
    time_elapsed = data.get('time_elapsed', 0)
    session_id = data.get('session_id')
    
    all_questions = question_manager.load_questions()
    
    correct = 0
    wrong_questions = []
    
    for ans in answers:
        question = next((q for q in all_questions if q['id'] == ans['question_id']), None)
        if question:
            if ans['answer'] == question['answer']:
                correct += 1
            else:
                wrong_questions.append({
                    'id': question['id'],
                    'question': question['question'],
                    'options': question['options'],
                    'user_answer': ans['answer'],
                    'correct_answer': question['answer'],
                    'category': question['category'],
                    'difficulty': question.get('difficulty', '中等')
                })
    
    total = len(answers)
    score = round(correct * 3, 1)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    analysis_result = {}
    weak_categories = []
    weak_groups = []
    group_weights = {}
    
    if session_id:
        answer_history = session_manager.get_answer_history(session_id)
        current_weights = session_manager.get_category_weights(session_id)
        
        analysis_result = local_optimizer.batch_analyze_and_update(
            answer_history,
            current_weights if current_weights else rf_selector.get_category_weights()
        )
        
        optimized_weights = analysis_result.get('weights', current_weights)
        rf_selector.set_category_weights(optimized_weights)
        session_manager.set_category_weights(session_id, optimized_weights)
        
        group_weights = analysis_result.get('group_weights', {})
        if group_weights:
            rf_selector.set_group_weights(group_weights)
        
        weak_categories = analysis_result.get('weak_categories', [])
        weak_groups = analysis_result.get('weak_groups', [])
    
    return jsonify({
        'success': True,
        'correct': correct,
        'total': total,
        'score': score,
        'accuracy': round(accuracy, 1),
        'time_elapsed': time_elapsed,
        'wrong_questions': wrong_questions,
        'category_analysis': analysis_result.get('category_analysis', {}),
        'group_analysis': analysis_result.get('group_analysis', {}),
        'weak_categories': weak_categories,
        'weak_groups': weak_groups,
        'group_weights': group_weights
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    all_questions = question_manager.load_questions()
    
    mastered = sum(1 for q in all_questions if q.get('status') == 'mastered')
    focus = sum(1 for q in all_questions if q.get('status') == 'focus')
    normal = sum(1 for q in all_questions if q.get('status', 'normal') == 'normal')
    
    categories = {}
    for q in all_questions:
        cat = q.get('category', '未知')
        categories[cat] = categories.get(cat, 0) + 1
    
    category_weights = weight_manager.get_category_weights()
    stats_summary = weight_manager.get_stats_summary()
    rf_weights = rf_selector.get_category_weights()
    
    return jsonify({
        'success': True,
        'total_questions': len(all_questions),
        'mastered': mastered,
        'focus': focus,
        'normal': normal,
        'categories': categories,
        'category_weights': category_weights,
        'rf_weights': rf_weights,
        'stats_summary': stats_summary
    })

@app.route('/api/weights/update', methods=['POST'])
def update_weights():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id:
        weights = session_manager.get_category_weights(session_id)
        if weights:
            rf_selector.set_category_weights(weights)
            return jsonify({'success': True, 'weights': weights})
    
    return jsonify({'success': False, 'error': '无权重数据'})

if __name__ == '__main__':
    session_manager.cleanup_old_sessions()
    app.run(debug=True, host='0.0.0.0', port=5000)
