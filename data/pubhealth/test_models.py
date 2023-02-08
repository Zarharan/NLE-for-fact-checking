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
@pytest.mark.parametrize("claim_id, main_text, summary, model_name",[(1, "This is the main text.", "The summary text.", "text-davinci-003")])
def test_insert_summary(text_summary_obj, claim_id, main_text, summary, model_name):
   ins_result= text_summary_obj.insert(claim_id, main_text, summary, model_name)
   assert ins_result.inserted_primary_key is not None


@pytest.mark.summaries
@pytest.mark.parametrize("claim_id",[(1)])
def test_select_summary(text_summary_obj, claim_id):
   select_result= text_summary_obj.select(claim_id)
   assert select_result is not None