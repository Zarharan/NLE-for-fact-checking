from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text


db_engine = create_engine('sqlite:///pubhealth.db', echo = True)


class TextSummary():

    def __init__(self):

        self.meta = MetaData()

        self.summaries = Table(
            'summaries', self.meta, 
            Column('id', Integer, primary_key = True), 
            Column('claim_id', Integer, unique=True), 
            Column('main_text', Text),
            Column('summary', Text),
            Column('model_name', String), # GPT3, Bart, etc
        )
    

    def create_table(self):
        self.meta.create_all(db_engine)


    def insert(self, claim_id, main_text, summary, model_name):
        ins_query = self.summaries.insert().values(claim_id = claim_id, main_text = main_text, summary= summary, model_name= model_name)
        conn = db_engine.connect()
        return conn.execute(ins_query)


    def select(self, claim_id):
        select_query = self.summaries.select().where(self.summaries.c.claim_id== claim_id)
        conn = db_engine.connect()
        return conn.execute(select_query)