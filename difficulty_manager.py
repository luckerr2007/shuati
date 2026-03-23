class DifficultyManager:
    def __init__(self):
        self.current_level = "中等"
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.total_correct = 0
        self.total_answered = 0
        self.levels = ["简单", "中等", "较难"]
    
    def record_correct(self):
        self.consecutive_correct += 1
        self.consecutive_wrong = 0
        self.total_correct += 1
        self.total_answered += 1
        self._adjust_difficulty()
    
    def record_wrong(self):
        self.consecutive_wrong += 1
        self.consecutive_correct = 0
        self.total_answered += 1
        self._adjust_difficulty()
    
    def _adjust_difficulty(self):
        if self.total_answered < 5:
            return
        
        accuracy = self.get_accuracy()
        
        if self.consecutive_correct >= 5 or accuracy > 80:
            self._increase_difficulty()
        elif self.consecutive_wrong >= 3 or accuracy < 50:
            self._decrease_difficulty()
    
    def _increase_difficulty(self):
        current_idx = self.levels.index(self.current_level)
        if current_idx < len(self.levels) - 1:
            self.current_level = self.levels[current_idx + 1]
            self.consecutive_correct = 0
    
    def _decrease_difficulty(self):
        current_idx = self.levels.index(self.current_level)
        if current_idx > 0:
            self.current_level = self.levels[current_idx - 1]
            self.consecutive_wrong = 0
    
    def get_current_level(self):
        return self.current_level
    
    def get_accuracy(self):
        if self.total_answered == 0:
            return 0
        return (self.total_correct / self.total_answered) * 100
    
    def get_difficulty_weights(self):
        level = self.current_level
        
        if level == "简单":
            return {"简单": 0.7, "中等": 0.25, "较难": 0.05}
        elif level == "中等":
            return {"简单": 0.3, "中等": 0.5, "较难": 0.2}
        else:
            return {"简单": 0.1, "中等": 0.4, "较难": 0.5}
    
    def reset(self):
        self.current_level = "中等"
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.total_correct = 0
        self.total_answered = 0
    
    def to_dict(self):
        return {
            "current_level": self.current_level,
            "consecutive_correct": self.consecutive_correct,
            "consecutive_wrong": self.consecutive_wrong,
            "total_correct": self.total_correct,
            "total_answered": self.total_answered,
            "accuracy": self.get_accuracy()
        }
    
    def from_dict(self, data):
        self.current_level = data.get("current_level", "中等")
        self.consecutive_correct = data.get("consecutive_correct", 0)
        self.consecutive_wrong = data.get("consecutive_wrong", 0)
        self.total_correct = data.get("total_correct", 0)
        self.total_answered = data.get("total_answered", 0)
