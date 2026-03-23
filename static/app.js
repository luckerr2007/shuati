const API_BASE = '';

class ExamApp {
    constructor() {
        this.questions = [];
        this.currentIndex = 0;
        this.answers = {};
        this.startTime = null;
        this.timerInterval = null;
        this.elapsedSeconds = 0;
        this.totalQuestions = 100;
        this.storageKey = 'exam_progress';
        this.sessionId = null;
        this.cultureScore = 200;
        this.lastYearCutoff = 390;
        this.expectedCultureScore = 230;
        this.expectedTotalScore = 430;
        
        this.initElements();
        this.bindEvents();
        this.loadStats();
        this.initSession();
    }
    
    initElements() {
        this.pages = {
            welcome: document.getElementById('welcome-page'),
            exam: document.getElementById('exam-page'),
            result: document.getElementById('result-page'),
            wrong: document.getElementById('wrong-page')
        };
        
        this.elements = {
            startBtn: document.getElementById('start-btn'),
            questionText: document.getElementById('question-text'),
            optionsContainer: document.getElementById('options-container'),
            difficultyTag: document.getElementById('difficulty-tag'),
            categoryTag: document.getElementById('category-tag'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            navGrid: document.getElementById('nav-grid'),
            timerDisplay: document.getElementById('timer'),
            questionProgress: document.getElementById('question-progress'),
            answeredDisplay: document.getElementById('answered-count'),
            difficultyIndicator: document.getElementById('difficulty-indicator'),
            prevBtn: document.getElementById('prev-btn'),
            nextBtn: document.getElementById('next-btn'),
            submitBtn: document.getElementById('submit-btn'),
            correctCount: document.getElementById('correct-count'),
            totalScore: document.getElementById('total-score'),
            accuracy: document.getElementById('accuracy'),
            timeUsed: document.getElementById('time-used'),
            resultEmoji: document.getElementById('result-emoji'),
            viewWrongBtn: document.getElementById('view-wrong-btn'),
            restartBtn: document.getElementById('restart-btn'),
            backResultBtn: document.getElementById('back-result-btn'),
            wrongList: document.getElementById('wrong-list'),
            totalQuestions: document.getElementById('total-questions'),
            masteredCount: document.getElementById('mastered-count'),
            focusCount: document.getElementById('focus-count'),
            cultureScoreInput: document.getElementById('culture-score'),
            cultureScoreDisplay: document.getElementById('culture-score-display'),
            skillScoreDisplay: document.getElementById('skill-score-display'),
            combinedScore: document.getElementById('combined-score'),
            comparisonResult: document.getElementById('comparison-result'),
            expectedAnalysis: document.getElementById('expected-analysis')
        };
    }
    
    bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this.startExam());
        this.elements.prevBtn.addEventListener('click', () => this.prevQuestion());
        this.elements.nextBtn.addEventListener('click', () => this.nextQuestion());
        this.elements.submitBtn.addEventListener('click', () => this.submitExam());
        this.elements.viewWrongBtn.addEventListener('click', () => this.showWrongQuestions());
        this.elements.restartBtn.addEventListener('click', () => this.restart());
        this.elements.backResultBtn.addEventListener('click', () => this.showPage('result'));
    }
    
    showPage(pageName) {
        Object.values(this.pages).forEach(page => page.classList.remove('active'));
        this.pages[pageName].classList.add('active');
    }
    
    async loadStats() {
        try {
            const response = await fetch(`${API_BASE}/api/stats`);
            const data = await response.json();
            
            if (data.success) {
                this.elements.totalQuestions.textContent = data.total_questions;
                this.elements.masteredCount.textContent = data.mastered;
                this.elements.focusCount.textContent = data.focus;
            }
        } catch (error) {
            console.error('加载统计失败:', error);
        }
    }
    
    async initSession() {
        const savedProgress = this.loadProgress();
        
        if (savedProgress && savedProgress.sessionId) {
            this.sessionId = savedProgress.sessionId;
            
            try {
                const response = await fetch(`${API_BASE}/api/session/restore`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: this.sessionId})
                });
                
                const data = await response.json();
                
                if (data.success && data.questions && data.questions.length > 0) {
                    this.showRestoreDialog(savedProgress, data);
                    return;
                }
            } catch (error) {
                console.error('恢复会话失败:', error);
            }
        }
        
        await this.createNewSession();
    }
    
    async createNewSession() {
        try {
            const response = await fetch(`${API_BASE}/api/session/create`, {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                this.sessionId = data.session_id;
                this.clearProgress();
            }
        } catch (error) {
            console.error('创建会话失败:', error);
        }
    }
    
    saveProgress() {
        const progressData = {
            sessionId: this.sessionId,
            answers: this.answers,
            currentIndex: this.currentIndex,
            questionIds: this.questions.map(q => q.id),
            startTime: this.startTime,
            elapsedSeconds: this.elapsedSeconds,
            timestamp: Date.now()
        };
        
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(progressData));
        } catch (error) {
            console.error('保存进度失败:', error);
        }
    }
    
    loadProgress() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (saved) {
                return JSON.parse(saved);
            }
        } catch (error) {
            console.error('加载进度失败:', error);
        }
        return null;
    }
    
    clearProgress() {
        try {
            localStorage.removeItem(this.storageKey);
        } catch (error) {
            console.error('清除进度失败:', error);
        }
    }
    
    showRestoreDialog(savedProgress, sessionData) {
        const dialog = document.createElement('div');
        dialog.className = 'restore-dialog';
        dialog.innerHTML = `
            <div class="restore-dialog-content">
                <div class="restore-dialog-title">发现未完成的答题进度</div>
                <div class="restore-dialog-info">
                    <p>已答题目: ${Object.keys(sessionData.answers || {}).length} / ${sessionData.total}</p>
                    <p>保存时间: ${new Date(savedProgress.timestamp).toLocaleString()}</p>
                </div>
                <div class="restore-dialog-buttons">
                    <button class="restore-btn restore-yes">恢复进度</button>
                    <button class="restore-btn restore-no">重新开始</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        dialog.querySelector('.restore-yes').addEventListener('click', () => {
            this.restoreProgress(savedProgress, sessionData);
            dialog.remove();
        });
        
        dialog.querySelector('.restore-no').addEventListener('click', async () => {
            await this.createNewSession();
            this.clearProgress();
            dialog.remove();
        });
    }
    
    restoreProgress(savedProgress, sessionData) {
        this.questions = sessionData.questions;
        this.answers = sessionData.answers || {};
        this.currentIndex = sessionData.current_index || 0;
        this.elapsedSeconds = savedProgress.elapsedSeconds || 0;
        this.startTime = savedProgress.startTime || Date.now();
        this.totalQuestions = this.questions.length;
        
        this.showPage('exam');
        this.renderQuestion();
        this.renderNavGrid();
        this.updateProgress();
        this.startTimer();
    }
    
    async startExam() {
        this.elements.startBtn.disabled = true;
        this.elements.startBtn.textContent = '正在加载题目...';
        
        try {
            const response = await fetch(`${API_BASE}/api/questions?count=${this.totalQuestions}&session_id=${this.sessionId}`);
            const data = await response.json();
            
            if (data.success && data.questions && data.questions.length > 0) {
                this.questions = data.questions;
                this.totalQuestions = data.questions.length;
                this.answers = {};
                this.currentIndex = 0;
                
                this.showPage('exam');
                this.renderQuestion();
                this.renderNavGrid();
                this.startTimer();
                this.saveProgress();
            } else {
                throw new Error(data.error || '加载题目失败');
            }
        } catch (error) {
            console.error('加载题目失败:', error);
            alert('加载题目失败，请检查网络连接');
        } finally {
            this.elements.startBtn.disabled = false;
            this.elements.startBtn.textContent = '开始答题';
        }
    }
    
    renderQuestion() {
        const question = this.questions[this.currentIndex];
        if (!question) return;
        
        this.elements.difficultyTag.textContent = question.difficulty;
        this.elements.difficultyTag.className = `difficulty-tag ${question.difficulty === '简单' ? 'easy' : question.difficulty === '中等' ? 'medium' : 'hard'}`;
        
        this.elements.categoryTag.textContent = question.category;
        
        this.elements.questionText.textContent = question.question;
        
        this.elements.optionsContainer.innerHTML = '';
        
        const options = question.options;
        const optionLabels = ['A', 'B', 'C', 'D'];
        
        optionLabels.forEach(label => {
            if (options[label]) {
                const optionDiv = document.createElement('div');
                optionDiv.className = 'option';
                optionDiv.dataset.option = label;
                
                if (this.answers[question.id] === label) {
                    optionDiv.classList.add('selected');
                }
                
                optionDiv.innerHTML = `
                    <span class="option-label">${label}</span>
                    <span class="option-text">${options[label]}</span>
                `;
                
                optionDiv.addEventListener('click', () => this.selectOption(question.id, label));
                this.elements.optionsContainer.appendChild(optionDiv);
            }
        });
        
        this.updateProgress();
        this.updateNavGrid();
    }
    
    async selectOption(questionId, option) {
        this.answers[questionId] = option;
        
        this.elements.optionsContainer.querySelectorAll('.option').forEach(opt => {
            opt.classList.remove('selected');
            if (opt.dataset.option === option) {
                opt.classList.add('selected');
            }
        });
        
        this.updateProgress();
        this.saveProgress();
        
        try {
            await fetch(`${API_BASE}/api/submit`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    question_id: questionId,
                    answer: option,
                    session_id: this.sessionId
                })
            });
        } catch (error) {
            console.error('提交答案失败:', error);
        }
    }
    
    updateProgress() {
        const answered = Object.keys(this.answers).length;
        const percent = (answered / this.totalQuestions) * 100;
        
        this.elements.progressFill.style.width = `${percent}%`;
        this.elements.progressText.textContent = `${answered} / ${this.totalQuestions}`;
        this.elements.questionProgress.textContent = `第 ${this.currentIndex + 1} 题`;
        this.elements.answeredDisplay.textContent = `已答 ${answered} 题`;
    }
    
    renderNavGrid() {
        this.elements.navGrid.innerHTML = '';
        
        for (let i = 0; i < this.questions.length; i++) {
            const btn = document.createElement('button');
            btn.className = 'nav-btn';
            btn.textContent = i + 1;
            btn.addEventListener('click', () => this.jumpToQuestion(i));
            this.elements.navGrid.appendChild(btn);
        }
        
        this.updateNavGrid();
    }
    
    updateNavGrid() {
        const btns = this.elements.navGrid.querySelectorAll('.nav-btn');
        btns.forEach((btn, index) => {
            btn.classList.remove('current', 'answered');
            
            if (index === this.currentIndex) {
                btn.classList.add('current');
            }
            
            if (this.answers[this.questions[index].id]) {
                btn.classList.add('answered');
            }
        });
    }
    
    jumpToQuestion(index) {
        this.currentIndex = index;
        this.saveProgress();
        this.renderQuestion();
    }
    
    prevQuestion() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.saveProgress();
            this.renderQuestion();
        }
    }
    
    nextQuestion() {
        if (this.currentIndex < this.questions.length - 1) {
            this.currentIndex++;
            this.saveProgress();
            this.renderQuestion();
        }
    }
    
    startTimer() {
        this.startTime = Date.now() - (this.elapsedSeconds * 1000);
        
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            this.elements.timerDisplay.textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
    
    async submitExam() {
        const answered = Object.keys(this.answers).length;
        
        if (answered < this.totalQuestions) {
            if (!confirm(`您还有 ${this.totalQuestions - answered} 题未作答，确定要交卷吗？`)) {
                return;
            }
        }
        
        clearInterval(this.timerInterval);
        
        const answers = Object.entries(this.answers).map(([questionId, answer]) => ({
            question_id: parseInt(questionId),
            answer: answer
        }));
        
        const timeElapsed = Math.floor((Date.now() - this.startTime) / 1000);
        
        try {
            const response = await fetch(`${API_BASE}/api/results`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    answers: answers,
                    time_elapsed: timeElapsed,
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showResults(data);
                this.clearProgress();
                await this.createNewSession();
            }
        } catch (error) {
            console.error('提交失败:', error);
            alert('提交失败，请重试');
        }
    }
    
    showResults(data) {
        const wrongCount = data.total - data.correct;
        const avgTime = data.total > 0 ? Math.round(data.time_elapsed / data.total) : 0;
        
        this.elements.correctCount.textContent = data.correct;
        document.getElementById('wrong-count').textContent = wrongCount;
        this.elements.totalScore.textContent = data.score;
        this.elements.accuracy.textContent = `${data.accuracy}%`;
        this.elements.timeUsed.textContent = this.formatTime(data.time_elapsed);
        document.getElementById('avg-time').textContent = `${avgTime}秒`;
        
        if (data.accuracy >= 80) {
            this.elements.resultEmoji.textContent = '🎉';
        } else if (data.accuracy >= 60) {
            this.elements.resultEmoji.textContent = '👍';
        } else {
            this.elements.resultEmoji.textContent = '💪';
        }
        
        this.wrongQuestions = data.wrong_questions || [];
        
        this.renderCategoryStats(data);
        this.renderDifficultyStats(data);
        this.renderWeakCategories(data);
        
        this.calculateAndDisplayTotalScore(data.score);
        
        this.showPage('result');
    }
    
    calculateAndDisplayTotalScore(skillScore) {
        const cultureScore = parseInt(this.elements.cultureScoreInput.value) || 200;
        this.cultureScore = cultureScore;
        
        const combinedTotal = cultureScore + skillScore;
        
        this.elements.cultureScoreDisplay.textContent = cultureScore;
        this.elements.skillScoreDisplay.textContent = skillScore;
        this.elements.combinedScore.textContent = combinedTotal;
        
        const diff = combinedTotal - this.lastYearCutoff;
        let comparisonHtml = '';
        
        if (diff >= 0) {
            comparisonHtml = `
                <div class="comparison-pass">
                    <span class="comparison-icon">✅</span>
                    <span class="comparison-text">恭喜！您的综合总分 <strong>${combinedTotal}分</strong> 超过去年分数线 <strong>${diff}分</strong></span>
                </div>
            `;
        } else {
            comparisonHtml = `
                <div class="comparison-fail">
                    <span class="comparison-icon">⚠️</span>
                    <span class="comparison-text">您的综合总分 <strong>${combinedTotal}分</strong> 低于去年分数线 <strong>${Math.abs(diff)}分</strong>，继续努力！</span>
                </div>
            `;
        }
        
        this.elements.comparisonResult.innerHTML = comparisonHtml;
        
        const expectedDiff = this.expectedTotalScore - combinedTotal;
        let expectedHtml = '';
        
        if (expectedDiff <= 0) {
            expectedHtml = `
                <div class="expected-pass">
                    <span class="expected-icon">🎉</span>
                    <span class="expected-text">恭喜达到预期！您的综合总分 <strong>${combinedTotal}分</strong> 已达到目标分数 <strong>430分</strong></span>
                </div>
            `;
        } else {
            const skillGap = expectedDiff;
            expectedHtml = `
                <div class="expected-fail">
                    <span class="expected-icon">📊</span>
                    <span class="expected-text">距离预期目标还差 <strong>${skillGap}分</strong>（目标：430分，当前：${combinedTotal}分）</span>
                    <div class="expected-hint">职业技能测试还需提高 ${skillGap} 分才能达到预期目标</div>
                </div>
            `;
        }
        
        this.elements.expectedAnalysis.innerHTML = expectedHtml;
    }
    
    renderCategoryStats(data) {
        const container = document.getElementById('category-stats');
        if (!container) return;
        
        const categoryAnalysis = data.category_analysis || {};
        const categories = Object.keys(categoryAnalysis);
        
        if (categories.length === 0) {
            container.innerHTML = '<p class="no-data">暂无分类统计数据</p>';
            return;
        }
        
        let html = '<div class="stats-grid-detailed">';
        
        categories.sort((a, b) => {
            const accA = categoryAnalysis[a].total > 0 ? categoryAnalysis[a].correct / categoryAnalysis[a].total : 0;
            const accB = categoryAnalysis[b].total > 0 ? categoryAnalysis[b].correct / categoryAnalysis[b].total : 0;
            return accA - accB;
        });
        
        for (const cat of categories) {
            const stats = categoryAnalysis[cat];
            const accuracy = stats.total > 0 ? Math.round(stats.correct / stats.total * 100) : 0;
            const barClass = accuracy >= 80 ? 'high' : accuracy >= 60 ? 'medium' : 'low';
            
            html += `
                <div class="stat-row">
                    <span class="stat-name">${cat}</span>
                    <div class="stat-bar-container">
                        <div class="stat-bar ${barClass}" style="width: ${accuracy}%"></div>
                    </div>
                    <span class="stat-percent">${accuracy}% (${stats.correct}/${stats.total})</span>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    renderDifficultyStats(data) {
        const container = document.getElementById('difficulty-stats');
        if (!container) return;
        
        const questions = this.questions;
        const answers = this.answers;
        
        const diffStats = {
            '简单': { correct: 0, total: 0 },
            '中等': { correct: 0, total: 0 },
            '较难': { correct: 0, total: 0 }
        };
        
        for (const q of questions) {
            const diff = q.difficulty || '中等';
            if (diffStats[diff]) {
                diffStats[diff].total++;
                if (answers[q.id] === q.answer) {
                    diffStats[diff].correct++;
                }
            }
        }
        
        let html = '<div class="stats-grid-detailed">';
        
        for (const [diff, stats] of Object.entries(diffStats)) {
            if (stats.total === 0) continue;
            
            const accuracy = Math.round(stats.correct / stats.total * 100);
            const barClass = accuracy >= 80 ? 'high' : accuracy >= 60 ? 'medium' : 'low';
            const diffClass = diff === '简单' ? 'easy' : diff === '中等' ? 'medium' : 'hard';
            
            html += `
                <div class="stat-row">
                    <span class="stat-name difficulty-tag ${diffClass}">${diff}</span>
                    <div class="stat-bar-container">
                        <div class="stat-bar ${barClass}" style="width: ${accuracy}%"></div>
                    </div>
                    <span class="stat-percent">${accuracy}% (${stats.correct}/${stats.total})</span>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    renderWeakCategories(data) {
        const container = document.getElementById('weak-categories');
        if (!container) return;
        
        const weakCategories = data.weak_categories || [];
        const weakGroups = data.weak_groups || [];
        
        if (weakCategories.length === 0 && weakGroups.length === 0) {
            container.innerHTML = '<p class="no-data">表现良好，无明显薄弱知识点！</p>';
            return;
        }
        
        let html = '';
        
        if (weakCategories.length > 0) {
            html += '<div class="weak-section"><h4>薄弱类别</h4><ul>';
            for (const cat of weakCategories.slice(0, 5)) {
                html += `<li class="weak-item-tag">${cat}</li>`;
            }
            html += '</ul></div>';
        }
        
        if (weakGroups.length > 0) {
            html += '<div class="weak-section"><h4>薄弱领域</h4><ul>';
            for (const group of weakGroups.slice(0, 3)) {
                html += `<li class="weak-item-tag">${group}</li>`;
            }
            html += '</ul></div>';
        }
        
        container.innerHTML = html;
    }
    
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${minutes}分${secs}秒`;
    }
    
    showWrongQuestions() {
        this.elements.wrongList.innerHTML = '';
        
        this.wrongQuestions.forEach((q, index) => {
            const item = document.createElement('div');
            item.className = 'wrong-item';
            item.innerHTML = `
                <div class="wrong-item-header">
                    <span class="wrong-item-number">${index + 1}</span>
                    <span class="wrong-item-category">${q.category}</span>
                </div>
                <div class="wrong-item-question">${q.question}</div>
                <div class="wrong-options">
                    ${Object.entries(q.options).map(([key, value]) => `
                        <div class="wrong-option ${key === q.correct_answer ? 'correct' : ''} ${key === q.user_answer && key !== q.correct_answer ? 'user-wrong' : ''}">
                            <span class="option-label">${key}</span>
                            <span class="option-text">${value}</span>
                        </div>
                    `).join('')}
                </div>
                <div class="wrong-answer-info">
                    <span class="user-answer">您的答案: ${q.user_answer}</span>
                    <span class="correct-answer">正确答案: ${q.correct_answer}</span>
                </div>
            `;
            this.elements.wrongList.appendChild(item);
        });
        
        this.showPage('wrong');
    }
    
    async restart() {
        await this.createNewSession();
        this.questions = [];
        this.answers = {};
        this.currentIndex = 0;
        this.elapsedSeconds = 0;
        this.clearProgress();
        this.loadStats();
        this.showPage('welcome');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.examApp = new ExamApp();
});
