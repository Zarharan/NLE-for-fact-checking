import pytest
from models import *


@pytest.fixture
def text_summary_obj():
   text_summary = TextSummary()
   return text_summary


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