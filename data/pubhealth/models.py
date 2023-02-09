from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import UniqueConstraint


Base = declarative_base()
db_engine = create_engine('mysql://factnle:nleF123456?@localhost/pubhealth', echo = False)


class SummaryModel(Base):

    __tablename__ = 'summaries'

    id = Column(Integer, primary_key = True)
    claim_id = Column(Integer, nullable=False)
    main_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    model_name= Column(String(32), nullable=False)
    UniqueConstraint("claim_id", "model_name" , name="unique_claim_id_model_name")
    
    
class TextSummary():

    def __init__(self):
        Session = sessionmaker(bind = db_engine)
        self.session = Session()


    def create_table(self):
        Base.metadata.create_all(db_engine)


    def insert(self, summary_data):
        self.session.add(summary_data)
        self.session.commit()
        return summary_data.id


    def select_summary(self, claim_id, model_name):
        
        select_result= self.session.query(SummaryModel).filter(SummaryModel.model_name== model_name,
            SummaryModel.claim_id== claim_id)
        
        if any(select_result):
            return select_result[0]
        
        return None