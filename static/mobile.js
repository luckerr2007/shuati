const API_BASE = '';

class MobileExamApp {
    constructor() {
        this.questions = [];
        this.currentIndex = 0;
        this.answers = {};
        this.startTime = null;
        this.timerInterval = null;
        this.elapsedSeconds = 0;
        this.totalQuestions = 100;
        this.storageKey = 'exam_progress_mobile';
        this.sessionId = null;
        this.cultureScore = 200;
        this.lastYearCutoff = 390;
        this.isNavDrawerOpen = false;
        this.isSubmitting = false;
        
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
            timerDisplay: document.getElementById('timer'),
            questionProgress: document.getElementById('question-progress'),
            answeredDisplay: document.getElementById('answered-count'),
            prevBtn: document.getElementById('prev-btn'),
            nextBtn: document.getElementById('next-btn'),
            submitBtn: document.getElementById('submit-btn'),
            correctCount: document.getElementById('correct-count'),
            totalScore: document.getElementById('total-score'),
            accuracy: document.getElementById('accuracy'),
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
            navDrawer: document.getElementById('nav-drawer'),
            navOverlay: document.getElementById('nav-overlay'),
            navToggle: document.getElementById('nav-toggle'),
            closeDrawer: document.getElementById('close-drawer'),
            navGrid: document.getElementById('nav-grid'),
            loadingOverlay: document.getElementById('loading-overlay'),
            categoryStats: document.getElementById('category-stats'),
            difficultyStats: document.getElementById('difficulty-stats'),
            weakCategories: document.getElementById('weak-categories')
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
        
        this.elements.navToggle.addEventListener('click', () => this.toggleNavDrawer());
        this.elements.closeDrawer.addEventListener('click', () => this.closeNavDrawer());
        this.elements.navOverlay.addEventListener('click', () => this.closeNavDrawer());
        
        this.elements.cultureScoreInput.addEventListener('change', () => {
            let val = parseInt(this.elements.cultureScoreInput.value) || 0;
            if (val > 300) val = 300;
            if (val < 0) val = 0;
            this.elements.cultureScoreInput.value = val;
            this.cultureScore = val;
        });
        
        let touchStartX = 0;
        document.addEventListener('touchstart', (e) => {
            touchStartX = e.touches[0].clientX;
        }, { passive: true });
        
        document.addEventListener('touchend', (e) => {
            if (!this.isNavDrawerOpen) return;
            const touchEndX = e.changedTouches[0].clientX;
            if (touchStartX - touchEndX > 50) {
                this.closeNavDrawer();
            }
        }, { passive: true });
    }
    
    toggleNavDrawer() {
        if (this.isNavDrawerOpen) {
            this.closeNavDrawer();
        } else {
            this.openNavDrawer();
        }
    }
    
    openNavDrawer() {
        this.elements.navDrawer.classList.add('open');
        this.elements.navOverlay.classList.add('show');
        this.isNavDrawerOpen = true;
    }
    
    closeNavDrawer() {
        this.elements.navDrawer.classList.remove('open');
        this.elements.navOverlay.classList.remove('show');
        this.isNavDrawerOpen = false;
    }
    
    showPage(pageName) {
        Object.values(this.pages).forEach(page => page.classList.remove('active'));
        this.pages[pageName].classList.add('active');
        window.scrollTo(0, 0);
    }
    
    showLoading(text = '加载中...') {
        this.elements.loadingOverlay.querySelector('.loading-text').textContent = text;
        this.elements.loadingOverlay.classList.add('show');
    }
    
    hideLoading() {
        this.elements.loadingOverlay.classList.remove('show');
    }
    
    showToast(message) {
        const existingToast = document.querySelector('.toast');
        if (existingToast) existingToast.remove();
        
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 2000);
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
            if (saved) return JSON.parse(saved);
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
        this.showToast('进度已恢复');
    }
    
    async startExam() {
        if (this.elements.startBtn.disabled) return;
        
        this.elements.startBtn.disabled = true;
        this.elements.startBtn.innerHTML = '<span class="btn-icon">⏳</span><span>加载中...</span>';
        
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
            this.showToast('加载题目失败，请检查网络');
        } finally {
            this.elements.startBtn.disabled = false;
            this.elements.startBtn.innerHTML = '<span class="btn-icon">▶</span><span>开始答题</span>';
        }
    }
    
    renderQuestion() {
        const question = this.questions[this.currentIndex];
        if (!question) return;
        
        const difficulty = question.difficulty || '中等';
        this.elements.difficultyTag.textContent = difficulty;
        this.elements.difficultyTag.className = `tag difficulty-tag ${difficulty === '简单' ? 'easy' : difficulty === '中等' ? 'medium' : 'hard'}`;
        
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
                    <span class="option-letter">${label}</span>
                    <span class="option-text">${options[label]}</span>
                `;
                
                optionDiv.addEventListener('click', () => this.selectOption(question.id, label));
                this.elements.optionsContainer.appendChild(optionDiv);
            }
        });
        
        this.updateProgress();
        this.updateNavGrid();
        this.updateNavigationButtons();
    }
    
    updateNavigationButtons() {
        this.elements.prevBtn.disabled = this.currentIndex === 0;
        this.elements.nextBtn.disabled = this.currentIndex === this.questions.length - 1;
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
        this.elements.questionProgress.textContent = `第${this.currentIndex + 1}题`;
        this.elements.answeredDisplay.textContent = `已答${answered}题`;
    }
    
    renderNavGrid() {
        this.elements.navGrid.innerHTML = '';
        
        for (let i = 0; i < this.questions.length; i++) {
            const btn = document.createElement('button');
            btn.className = 'nav-item';
            btn.textContent = i + 1;
            btn.addEventListener('click', () => {
                this.jumpToQuestion(i);
                this.closeNavDrawer();
            });
            this.elements.navGrid.appendChild(btn);
        }
        
        this.updateNavGrid();
    }
    
    updateNavGrid() {
        const btns = this.elements.navGrid.querySelectorAll('.nav-item');
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
            
            this.elements.timerDisplay.classList.remove('warning', 'danger');
            if (elapsed >= 5400) {
                this.elements.timerDisplay.classList.add('danger');
            } else if (elapsed >= 4500) {
                this.elements.timerDisplay.classList.add('warning');
            }
        }, 1000);
    }
    
    async submitExam() {
        if (this.isSubmitting) return;
        
        const answered = Object.keys(this.answers).length;
        
        if (answered < this.totalQuestions) {
            if (!confirm(`您还有 ${this.totalQuestions - answered} 题未作答，确定要交卷吗？`)) {
                return;
            }
        }
        
        this.isSubmitting = true;
        clearInterval(this.timerInterval);
        this.showLoading('正在提交...');
        
        const answers = Object.entries(this.answers).map(([questionId, answer]) => ({
            question_id: parseInt(questionId),
            answer: answer
        }));
        
        const timeElapsed = Math.floor((Date.now() - this.startTime) / 1000);
        
        try {
            const response = await fetch(`${API_BASE}/api/results`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
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
            this.showToast('提交失败，请重试');
        } finally {
            this.hideLoading();
            this.isSubmitting = false;
        }
    }
    
    showResults(data) {
        const wrongCount = data.total - data.correct;
        
        this.elements.correctCount.textContent = data.correct;
        document.getElementById('wrong-count').textContent = wrongCount;
        this.elements.totalScore.textContent = data.score;
        this.elements.accuracy.textContent = `${data.accuracy}%`;
        
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
                    <span>✅ 恭喜！综合总分 <strong>${combinedTotal}分</strong></span>
                    <span>超过去年分数线 <strong>${diff}分</strong></span>
                </div>
            `;
        } else {
            comparisonHtml = `
                <div class="comparison-fail">
                    <span>⚠️ 综合总分 <strong>${combinedTotal}分</strong></span>
                    <span>低于去年分数线 <strong>${Math.abs(diff)}分</strong></span>
                </div>
            `;
        }
        
        this.elements.comparisonResult.innerHTML = comparisonHtml;
    }
    
    renderCategoryStats(data) {
        const container = this.elements.categoryStats;
        if (!container) return;
        
        const categoryAnalysis = data.category_analysis || {};
        const categories = Object.keys(categoryAnalysis);
        
        if (categories.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); font-size: 14px;">暂无数据</p>';
            return;
        }
        
        let html = '';
        
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
                    <div class="stat-name">${cat} (${stats.correct}/${stats.total})</div>
                    <div class="stat-bar">
                        <div class="stat-bar-fill ${barClass}" style="width: ${accuracy}%"></div>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }
    
    renderDifficultyStats(data) {
        const container = this.elements.difficultyStats;
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
        
        let html = '';
        
        for (const [diff, stats] of Object.entries(diffStats)) {
            if (stats.total === 0) continue;
            
            const accuracy = Math.round(stats.correct / stats.total * 100);
            const barClass = accuracy >= 80 ? 'high' : accuracy >= 60 ? 'medium' : 'low';
            
            html += `
                <div class="stat-row">
                    <div class="stat-name">${diff} (${stats.correct}/${stats.total})</div>
                    <div class="stat-bar">
                        <div class="stat-bar-fill ${barClass}" style="width: ${accuracy}%"></div>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html || '<p style="color: var(--text-secondary); font-size: 14px;">暂无数据</p>';
    }
    
    renderWeakCategories(data) {
        const container = this.elements.weakCategories;
        if (!container) return;
        
        const weakCategories = data.weak_categories || [];
        const weakGroups = data.weak_groups || [];
        
        if (weakCategories.length === 0 && weakGroups.length === 0) {
            container.innerHTML = '<p style="color: var(--success-color); font-size: 14px;">表现良好，无明显薄弱点！</p>';
            return;
        }
        
        let html = '<div class="weak-tags">';
        
        for (const cat of weakCategories.slice(0, 5)) {
            html += `<span class="weak-tag">${cat}</span>`;
        }
        
        for (const group of weakGroups.slice(0, 3)) {
            html += `<span class="weak-tag">${group}</span>`;
        }
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    showWrongQuestions() {
        this.elements.wrongList.innerHTML = '';
        
        if (this.wrongQuestions.length === 0) {
            this.elements.wrongList.innerHTML = `
                <div class="no-wrong">
                    <h3>🎉 全部正确！</h3>
                    <p>太棒了，没有错题</p>
                </div>
            `;
        } else {
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
                                <span class="option-letter">${key}</span>
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
        }
        
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
    window.mobileExamApp = new MobileExamApp();
});
