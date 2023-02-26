import pytest
from models import *


@pytest.fixture
def text_summary_obj():
   text_summary = TextSummary()
   return text_summary


@pytest.fixture
def experiment_obj():
   experiment = Experiments()
   return experiment


# using pytest -m summaries -v
@pytest.mark.summaries
def test_create_summary_table(text_summary_obj):
   ins_result= text_summary_obj.create_table()
   assert True


@pytest.mark.summaries
@pytest.mark.parametrize("claim_id, main_text, summary, model_name",[(1, "This is the main text.", "The summary text.", "bart")])
def test_insert_summary(text_summary_obj, claim_id, main_text, summary, model_name):
   summary_data= SummaryModel(claim_id= claim_id, main_text= main_text, summary= summary
      , model_name= model_name)
   assert text_summary_obj.insert(summary_data) > 0


@pytest.mark.summaries
@pytest.mark.parametrize("claim_id, model_name",[(1, "bart")])
def test_select_summary(text_summary_obj, claim_id, model_name):
   select_result= text_summary_obj.select_summary(claim_id, model_name)
   assert select_result is not None


# using pytest -m experiments -v
@pytest.mark.experiments
def test_create_experiment_table(experiment_obj):
   ins_result= experiment_obj.create_table()
   assert True   


@pytest.mark.experiments
@pytest.mark.parametrize("args, args_hash",[("{'summarize': 'False', 'seed': '313'}", "cfba7b4862252f33a7d497e73824f99c4646cf21f2b318a9f21284d44825f912")])
def test_insert_experiment(experiment_obj, args, args_hash):
   experiment_data= ExperimentModel(args= args, args_hash= args_hash)
   assert experiment_obj.insert(experiment_data) > 0


@pytest.mark.experiments
@pytest.mark.parametrize("args_hash",[("cfba7b4862252f33a7d497e73824f99c4646cf21f2b318a9f21284d44825f912")])
def test_select_experiment(experiment_obj, args_hash):
   select_result= experiment_obj.select_experiment(args_hash)
   assert select_result is not None