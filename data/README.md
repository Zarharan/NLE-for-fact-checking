# Explored Datasets

## PUBHEALTH fact-checking dataset

The PUBHEALTH dataset provides automated fact-checking of public health claims that are explainable. In the PUBHEALTH dataset, each instance has a veracity label (true, false, unproven, mixture). Additionally, each instance in the dataset has an explanation text field. The explanation is a justification for why a particular veracity label has been given to the claim.
For more information check the source of the dataset [here](https://github.com/neemakot/Health-Fact-Checking).

### Average Token Counts in PUBHEALTH dataset
When we examine the average number of tokens in the dev set and test set of the PUBHEALTH dataset, we find that both zero-shot and few-shot require the context of instances to be summarized. The following images show the average number of tokens in the dev set and test set respectively.

![The average number of tokens in the dev set](https://github.com/Zarharan/NLE-for-fact-checking/blob/main/data/pubhealth/dev_avg_no_tokens.png)

![The average number of tokens in the test set](https://github.com/Zarharan/NLE-for-fact-checking/blob/main/data/pubhealth/test_avg_no_tokens.png)