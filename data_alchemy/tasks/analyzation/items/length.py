class ContextLengthAnalyzer:
    def __init__(self, context):
        self.context = context

    def analyze(self):
        return len(self.context)