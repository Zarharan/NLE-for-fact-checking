class NLEMetrics():
    # با فایل های بالا روی درایو متریک ها رو تست کن و بزا
    # ساخت دیتابیس کانفیگ ورودی
    # متریک ها رو کامل کن و روال کلی رو نگاه بنداز که بهینه باشه و توضیحات کلاس ها رو چک کن

    def __init__(self):
        self.avg_rouge_result= {'rouge1':{'pre':0, 'rec': 0, 'f1':0}
            , 'rouge2':{'pre':0, 'rec': 0, 'f1':0}, 'rougeL': {'pre':0, 'rec': 0, 'f1':0}}
        self.scorer = rouge_scorer.RougeScorer(self.avg_rouge_result.keys(), use_stemmer=True)


    def avg_rouge_scores(self, first_list, second_list):
        
        for f_text, s_text in zip(first_list, second_list):
            rouge_score= self.scorer.score(f_text, s_text)
            for key in rouge_score.keys():
                self.avg_rouge_result[key]["pre"]+= rouge_score[key].precision
                self.avg_rouge_result[key]["rec"]+= rouge_score[key].recall
                self.avg_rouge_result[key]["f1"]+= rouge_score[key].fmeasure

        return self.avg_rouge_resul

    
    def avg_snli_scores(self):
        pass