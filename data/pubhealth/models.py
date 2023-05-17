from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, ForeignKey, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import relationship
import datetime


Base = declarative_base()
db_engine = create_engine('mysql://factnle:nleF123456?@localhost/pubhealth', echo = False)


class SummaryModel(Base):
    '''
    The SummaryModel object is a model of a record from summaries table in the database   
    '''

    __tablename__ = 'summaries'
    # create a unique key for claim_id model_name pairs
    __table_args__ = (
        UniqueConstraint("claim_id", "model_name" , name="unique_claim_id_model_name"),
    )

    id = Column(Integer, primary_key = True)
    claim_id = Column(Integer, nullable=False)
    main_text = Column(Text, nullable=False) # Main text of an instance
    summary = Column(Text, nullable=False) # summarized text of the main text
    model_name= Column(String(32), nullable=False) # the model name that summarized the main text
    
    # UniqueConstraint("claim_id", "model_name" , name="unique_claim_id_model_name")


class ExperimentModel(Base):
    '''
    The ExperimentModel object is a model of a record from experiments table in the database
    '''

    __tablename__ = 'experiments'

    id = Column(Integer, primary_key = True)
    args = Column(Text, nullable=False) # The keys and values of the input arguments for an experiment
    args_hash = Column(String(64), unique=True, nullable=False) # The Hash of the args
    completed = Column(Boolean, nullable=False, default= True) # Shows whether the experiment was completed or not
    insert_date= Column(DateTime, nullable=True, default=datetime.datetime.now)
    results = relationship("ExperimentResultModel", back_populates = "experiment")
    experiment_instances = relationship("ExperimentInstancesModel", back_populates = "experiment")
    

class ExperimentResultModel(Base):
    '''
    The ExperimentResultModel object is a model of a record from results table that saves result of experiments in the database
    '''

    __tablename__ = 'results'

    id = Column(Integer, primary_key = True)
    experiment_id = Column(Integer, ForeignKey('experiments.id')) # The id of related experiment which this record belongs to
    file_path = Column(Text, nullable=False) # The path of result file of an experiment
    # create the relationship between experiments and results
    experiment = relationship("ExperimentModel", back_populates = "results")


class ExperimentInstancesModel(Base):
    '''
    The ExperimentInstancesModel object is a model of a record from experiment_instances table that saves result of experiments for each instances in the database
    '''

    __tablename__ = 'experiment_instances'

    id = Column(Integer, primary_key = True)
    experiment_id = Column(Integer, ForeignKey('experiments.id')) # The id of related experiment which this record belongs to
    claim_id = Column(Integer, nullable=False) # The Id of the related claim to the instance
    result = Column(Text, nullable=False) # The result (predicted veracity, generated explanation, or both) of each instances in the experiment
    # create the relationship between experiments and results
    experiment = relationship("ExperimentModel", back_populates = "experiment_instances")    


class TextSummary():
    '''
    The TextSummary object is responsible for creation of the summaries table and its CRUD operation.

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
        ''' This function select a record from the summaries table by claim_id and model_name.

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


class Experiments():
    '''
    The Experiments object is responsible for creation of the experiments table and its CRUD operation.

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


    def insert(self, experiment_data):
        ''' This function add an object to the experiments table.

        :param experiment_data: The object that contains experiment data
        :type experiment_data: object

        :returns: The ID of the created record
        :rtype: int
        '''
        self.session.add(experiment_data)
        self.session.commit()
        return experiment_data.id


    def insert_result(self, result_data):
        ''' This function add the result of an experiment to the results table.

        :param result_data: The object that contains results information of the experiment
        :type result_data: object

        :returns: The ID of the created record
        :rtype: int
        '''
        self.session.add(result_data)
        self.session.commit()
        return result_data.id


    def insert_instances(self, instance_result_data):
        ''' This function add the result of each instances of an experiment to the related table.

        :param instance_result_data: The object that contains results information of the experiment
        :type instance_result_data: object

        :returns: The ID of the created record
        :rtype: int
        '''
        self.session.add(instance_result_data)
        self.session.commit()
        return instance_result_data.id        


    def select_experiment(self, args_hash):
        ''' This function select a record from the experiments table by args_hash.

        :param args_hash: The Hash of the input args for an experiment
        :type args_hash: str

        :returns: The experiment object of the selected record
        :rtype: object
        '''

        select_result= self.session.query(ExperimentModel).join(ExperimentResultModel).filter(ExperimentModel.args_hash== args_hash)
        
        if any(select_result):
            return select_result[0]
        
        return None       