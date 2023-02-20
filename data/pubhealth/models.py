from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import UniqueConstraint


Base = declarative_base()
db_engine = create_engine('mysql://factnle:nleF123456?@localhost/pubhealth', echo = False)


class SummaryModel(Base):
    '''
    The SummaryModel object is a model of a record from summaries table in the database   
    '''

    __tablename__ = 'summaries'

    id = Column(Integer, primary_key = True)
    claim_id = Column(Integer, nullable=False)
    main_text = Column(Text, nullable=False) # Main text of an instance
    summary = Column(Text, nullable=False) # summarized text of the main text
    model_name= Column(String(32), nullable=False) # the model name that summarized the main text
    # create a unique key for claim_id model_name pairs
    UniqueConstraint("claim_id", "model_name" , name="unique_claim_id_model_name")
    
    
class TextSummary():
    '''
    The TextSummary object is responsible for creation of the summaries table and CRUD operation.

    :ivar session: A session object is the handle to database.
    :vartype session: object        
    '''
    def __init__(self):
        Session = sessionmaker(bind = db_engine)
        self.session = Session()


    def create_table(self):
        ''' This function creates all tables that have not been created yet.

        :returns: Nothing
        :rtype: None
        '''        
        Base.metadata.create_all(db_engine)


    def insert(self, summary_data):
        ''' This function add an object to the summaries table.

        :param summary_data: The object that contains summary data
        :type summary_data: object

        :returns: The ID of the created record
        :rtype: int
        '''
        self.session.add(summary_data)
        self.session.commit()
        return summary_data.id


    def select_summary(self, claim_id, model_name):
        ''' This function add an object to the summaries table.

        :param claim_id: The related claim ID to filter out a record
        :type claim_id: int
        :param model_name: The model name that summarized the main text to filter out a record
        :type model_name: str        

        :returns: The summary object of the selected record
        :rtype: object
        '''

        select_result= self.session.query(SummaryModel).filter(SummaryModel.model_name== model_name,
            SummaryModel.claim_id== claim_id)
        
        if any(select_result):
            return select_result[0]
        
        return None