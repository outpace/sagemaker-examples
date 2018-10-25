
# Spotlight Implicit Factorization

An implicit feedback matrix factorization model. Uses a classic matrix factorization approach, with latent vectors used to represent both users and items. Their dot product gives the predicted score for a user-item pair.

[Spotlight Documentation](https://maciejkula.github.io/spotlight/factorization/implicit.html)

## Table of contents

* [Sample files](#sample-files)
* [Step 1 - Prepare training data](#prepare-training-data)
 * [Download movielens 100k dataset](#download-movielens)
 * [Import ratings data](#import-ratings-data)
 * [Create training data file](#create-training-data-file)
 * [Upload training data file](#upload-training-data)
* [Step 2 - Create a model](#create-model)
 * [Run a SageMaker training job](#run-training-job)
 * [Create a SageMaker model](#create-sagemaker-model)
* [Step 3 - Get recommendations (inference)](#get-recommendations)
 * [Create batch transform input file](#create-batch-input)
 * [Upload the batch transform input file to s3](#upload-batch-input)
 * [Download the batch results](#download-batch-results)
 * [Import movie titles](#import-movie-titles)
 * [Recommendations with scores](#recommendations)
 * [User history](#user-history)
* [Step 4 - Optional Cleanup](#cleanup)

## Sample files <a id="sample-files"></a>

These links will work after running all the code in this notebook.

* [training input file](https://github.com/outpace/sagemaker-examples/blob/master/train_data/ml-100k-gt2.csv)
* [batch transform input file](https://github.com/outpace/sagemaker-examples/blob/master/batch_input/recommendation.requests)
* [batch transform output file](https://github.com/outpace/sagemaker-examples/blob/master/recommendation.requests.out)

## Step 1 - Prepare training data <a id="prepare-training-data"></a>
### Download movielens 100k dataset <a id="download-movielens"></a>


```python
!wget --no-clobber http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip -o ml-100k.zip
```

    --2018-10-24 23:01:36--  http://files.grouplens.org/datasets/movielens/ml-100k.zip
    Resolving files.grouplens.org (files.grouplens.org)... 128.101.34.235
    Connecting to files.grouplens.org (files.grouplens.org)|128.101.34.235|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4924029 (4.7M) [application/zip]
    Saving to: ‘ml-100k.zip’
    
    ml-100k.zip         100%[===================>]   4.70M  15.2MB/s    in 0.3s    
    
    2018-10-24 23:01:36 (15.2 MB/s) - ‘ml-100k.zip’ saved [4924029/4924029]
    
    Archive:  ml-100k.zip
       creating: ml-100k/
      inflating: ml-100k/allbut.pl       
      inflating: ml-100k/mku.sh          
      inflating: ml-100k/README          
      inflating: ml-100k/u.data          
      inflating: ml-100k/u.genre         
      inflating: ml-100k/u.info          
      inflating: ml-100k/u.item          
      inflating: ml-100k/u.occupation    
      inflating: ml-100k/u.user          
      inflating: ml-100k/u1.base         
      inflating: ml-100k/u1.test         
      inflating: ml-100k/u2.base         
      inflating: ml-100k/u2.test         
      inflating: ml-100k/u3.base         
      inflating: ml-100k/u3.test         
      inflating: ml-100k/u4.base         
      inflating: ml-100k/u4.test         
      inflating: ml-100k/u5.base         
      inflating: ml-100k/u5.test         
      inflating: ml-100k/ua.base         
      inflating: ml-100k/ua.test         
      inflating: ml-100k/ub.base         
      inflating: ml-100k/ub.test         


### Import ratings data <a id="import-ratings-data"></a>

Keep only ratings strictly higher than 2 to make this an implicit dataset.


```python
import pandas as pd

implicit_df = pd.read_csv('ml-100k/u.data', sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])
implicit_df = implicit_df[implicit_df.rating>2][["user_id", "item_id"]]
implicit_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
    </tr>
    <tr>
      <th>5</th>
      <td>298</td>
      <td>474</td>
    </tr>
    <tr>
      <th>7</th>
      <td>253</td>
      <td>465</td>
    </tr>
    <tr>
      <th>8</th>
      <td>305</td>
      <td>451</td>
    </tr>
  </tbody>
</table>
</div>



### Create training data file <a id="create-training-data-file"></a>

Create a csv file from the dataframe above. Do not include the index, but include headers `user_id` and `item_id`.


```python
from IPython.display import HTML

TRAIN_DATA_DIR = 'train_data'

!mkdir -p {TRAIN_DATA_DIR}
implicit_df.to_csv('{}/ml-100k-gt2.csv'.format(TRAIN_DATA_DIR), index=False)

HTML('After you run this block, click <a href="https://github.com/outpace/sagemaker-examples/blob/master/%s/ml-100k-gt2.csv" target="_blank">here</a> to see what the training data file looks like.'%(TRAIN_DATA_DIR))
```




After you run this block, click <a href="https://github.com/outpace/sagemaker-examples/blob/master/train_data/ml-100k-gt2.csv" target="_blank">here</a> to see what the training data file looks like.



### Upload training data to s3 <a id="upload-training-data"></a>

Choose a bucket, optionally customize the prefix, and upload the csv created above.


```python
import sagemaker as sage

bucket = "sagemaker-validation-us-east-2"
prefix = "spotlight-implicit-factorization-test"

sess = sage.Session()

s3_train = sess.upload_data(TRAIN_DATA_DIR, bucket, "{}/training".format(prefix))
"uploaded training data file to {}".format(s3_train)
```




    'uploaded training data file to s3://sagemaker-validation-us-east-2/spotlight-implicit-factorization-test/training'



## Step 2 - Create a model <a id="create-model"></a>

### Run a SageMaker training job <a id="run-training-job"></a>

This code will start a training job, wait for it to be done, and report its status.


```python
%%time

import boto3
import time
from sagemaker import get_execution_role

role = get_execution_role()
ecr_image = "435525115971.dkr.ecr.us-east-2.amazonaws.com/sagemaker/spotlight-implicit:76"
job_name_prefix = 'spotlight-implicit-factorization-test'
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
job_name = job_name_prefix + timestamp
create_training_params = \
{
    "AlgorithmSpecification": {
        "TrainingImage": ecr_image,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": 's3://{}/{}/output'.format(bucket, job_name_prefix)
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.p3.2xlarge",
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": job_name,
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 360000
    },
    "InputDataConfig": [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_train,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/csv",
            "CompressionType": "None"
        }
    ]
}

sagemaker = boto3.client(service_name='sagemaker')
sagemaker.create_training_job(**create_training_params)
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    job_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = job_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))
```

    Training job current status: InProgress
    Training job ended with status: Completed
    CPU times: user 150 ms, sys: 14.8 ms, total: 165 ms
    Wall time: 4min


### Create a SageMaker model <a id="create-sagemaker-model"></a>

This will set up the model created during training within SageMaker to be used later for recommendations.


```python
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
model_name="spotlight-implicit-factorization-test" + timestamp
job_info = sagemaker.describe_training_job(TrainingJobName=job_name)
model_data = job_info['ModelArtifacts']['S3ModelArtifacts']

primary_container = {
    'Image': ecr_image,
    'ModelDataUrl': model_data,
}

create_model_response = sagemaker.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    PrimaryContainer = primary_container)

create_model_response
```




    {'ModelArn': 'arn:aws:sagemaker:us-east-2:435525115971:model/spotlight-implicit-factorization-test-2018-10-24-23-05-39',
     'ResponseMetadata': {'RequestId': '41855115-8989-4b29-bc63-c2f417843b83',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'x-amzn-requestid': '41855115-8989-4b29-bc63-c2f417843b83',
       'content-type': 'application/x-amz-json-1.1',
       'content-length': '119',
       'date': 'Wed, 24 Oct 2018 23:05:39 GMT'},
      'RetryAttempts': 0}}



## Step 3 - Get recommendations (Inference) <a id="get-recommendations"></a>

### Create batch transform input file <a id="create-batch-input"></a>

Each row is a json object containing two keys:

* `user_id`: the id of the user to get recommendations for
* `top_n`: the number of top scoring recommendations to return


```python
import json

BATCH_INPUT_DIR = 'batch_input'

!mkdir -p {BATCH_INPUT_DIR}

with open(BATCH_INPUT_DIR + '/recommendation.requests', 'w') as outfile:
    json.dump({"user_id": "685", "top_n": "5"}, outfile)
    outfile.write("\n")
    json.dump({"user_id": "302", "top_n": "5"}, outfile)
    
HTML('After you run this block, click <a href="https://github.com/outpace/sagemaker-examples/blob/master/%s/recommendation.requests" target="_blank">here</a> to see what the batch input file looks like.'%(BATCH_INPUT_DIR))
```




After you run this block, click <a href="https://github.com/outpace/sagemaker-examples/blob/master/batch_input/recommendation.requests" target="_blank">here</a> to see what the batch input file looks like.



### Upload the batch transform input file to s3 <a id="upload-batch-input"></a>


```python
batch_input = sess.upload_data(BATCH_INPUT_DIR, bucket, "{}/batch_input".format(prefix))
"uploaded training data file to {}".format(batch_input)
```




    'uploaded training data file to s3://sagemaker-validation-us-east-2/spotlight-implicit-factorization-test/batch_input'



### Run the Batch Transform Job

This code will start a batch transform job, wait for it to be done, and report its status.


```python
%%time

timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
batch_job_name = "spotlight-implicit-factorization-test" + timestamp
batch_output = 's3://{}/{}/output'.format(bucket, batch_job_name)
request = \
{
  "TransformJobName": batch_job_name,
  "ModelName": model_name,
  "BatchStrategy": "SingleRecord",
  "TransformInput": {
    "DataSource": {
      "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": batch_input
      }
    },
    "ContentType": "application/json",
    "CompressionType": "None",
    "SplitType": "Line"
  },
  "TransformOutput": {
    "S3OutputPath": batch_output,
    "Accept": "text/csv",
    "AssembleWith": "Line"
  },
  "TransformResources": {
    "InstanceType": "ml.p3.2xlarge",
    "InstanceCount": 1
  }
}

sagemaker.create_transform_job(**request)

print("Created Transform job with name: ", batch_job_name)

while(True):
    job_info = sagemaker.describe_transform_job(TransformJobName=batch_job_name)
    status = job_info['TransformJobStatus']
    if status == 'Completed':
        print("Transform job ended with status: " + status)
        break
    if status == 'Failed':
        message = job_info['FailureReason']
        print('Transform failed with the following error: {}'.format(message))
        raise Exception('Transform job failed') 
    time.sleep(30)
```

    Created Transform job with name:  spotlight-implicit-factorization-test-2018-10-24-23-05-39
    Transform job ended with status: Completed
    CPU times: user 113 ms, sys: 3.76 ms, total: 117 ms
    Wall time: 4min 31s


### Download the batch results <a id="download-batch-results"></a>


```python
!aws s3 cp {batch_output + '/recommendation.requests.out'} .

HTML('After you run this block, click <a href="https://github.com/outpace/sagemaker-examples/blob/master/recommendation.requests.out" target="_blank">here</a> to see what the batch output file looks like.')
```

    download: s3://sagemaker-validation-us-east-2/spotlight-implicit-factorization-test-2018-10-24-23-05-39/output/recommendation.requests.out to ./recommendation.requests.out





After you run this block, click <a href="https://github.com/outpace/sagemaker-examples/blob/master/recommendation.requests.out" target="_blank">here</a> to see what the batch output file looks like.



### Import movie titles <a id="import-movie-titles"></a>

Get movie titles in `u.item` from the movielens files downloaded earlier and join with ratings data.


```python
titles_df = pd.read_csv('ml-100k/u.item', sep="|", header=None, encoding = "ISO-8859-1").set_index([0]).iloc[:,0:1]
implicit_df = implicit_df.join(titles_df, on='item_id').rename(index=str, columns={"user_id":"user_id",1:"movie_title"})
implicit_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>movie_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>L.A. Confidential (1997)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>298</td>
      <td>474</td>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>253</td>
      <td>465</td>
      <td>Jungle Book, The (1994)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>305</td>
      <td>451</td>
      <td>Grease (1978)</td>
    </tr>
  </tbody>
</table>
</div>



### Recommendations with scores <a id="recommendations"></a>

Import the recommendations from the batch output file downloaded above and join with titles dataframe. These are the top 5 movie recommendations for users 685 and 302.


```python
recommendations_df = pd.read_csv('recommendation.requests.out', header=None, names=["user_id", "item_id", "score"])
recommendations_df = recommendations_df.join(titles_df, on='item_id').rename(index=str, columns={"user_id":"user_id",1:"movie_title"})
recommendations_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>score</th>
      <th>movie_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>685</td>
      <td>333</td>
      <td>14.640368</td>
      <td>Game, The (1997)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>685</td>
      <td>272</td>
      <td>13.269605</td>
      <td>Good Will Hunting (1997)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>685</td>
      <td>268</td>
      <td>13.025604</td>
      <td>Chasing Amy (1997)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>685</td>
      <td>347</td>
      <td>12.637764</td>
      <td>Wag the Dog (1997)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>685</td>
      <td>315</td>
      <td>12.535725</td>
      <td>Apt Pupil (1998)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>302</td>
      <td>748</td>
      <td>12.058822</td>
      <td>Saint, The (1997)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>302</td>
      <td>333</td>
      <td>11.834901</td>
      <td>Game, The (1997)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>302</td>
      <td>323</td>
      <td>11.719286</td>
      <td>Dante's Peak (1997)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>302</td>
      <td>258</td>
      <td>10.959321</td>
      <td>Contact (1997)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>302</td>
      <td>313</td>
      <td>10.367278</td>
      <td>Titanic (1997)</td>
    </tr>
  </tbody>
</table>
</div>



### User history <a id="user-history"></a>

For reference, here are the movies users 685 and 302 watched/rated.


```python
implicit_df[implicit_df.user_id.isin([685,302])].sort_values(by=['user_id'], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>movie_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42723</th>
      <td>685</td>
      <td>269</td>
      <td>Full Monty, The (1997)</td>
    </tr>
    <tr>
      <th>43618</th>
      <td>685</td>
      <td>302</td>
      <td>L.A. Confidential (1997)</td>
    </tr>
    <tr>
      <th>53414</th>
      <td>685</td>
      <td>325</td>
      <td>Crash (1996)</td>
    </tr>
    <tr>
      <th>70388</th>
      <td>685</td>
      <td>324</td>
      <td>Lost Highway (1997)</td>
    </tr>
    <tr>
      <th>89437</th>
      <td>685</td>
      <td>882</td>
      <td>Washington Square (1997)</td>
    </tr>
    <tr>
      <th>98163</th>
      <td>685</td>
      <td>875</td>
      <td>She's So Lovely (1997)</td>
    </tr>
    <tr>
      <th>4826</th>
      <td>302</td>
      <td>328</td>
      <td>Conspiracy Theory (1997)</td>
    </tr>
    <tr>
      <th>9848</th>
      <td>302</td>
      <td>307</td>
      <td>Devil's Advocate, The (1997)</td>
    </tr>
    <tr>
      <th>14758</th>
      <td>302</td>
      <td>258</td>
      <td>Contact (1997)</td>
    </tr>
    <tr>
      <th>32327</th>
      <td>302</td>
      <td>301</td>
      <td>In &amp; Out (1997)</td>
    </tr>
    <tr>
      <th>55280</th>
      <td>302</td>
      <td>358</td>
      <td>Spawn (1997)</td>
    </tr>
    <tr>
      <th>58767</th>
      <td>302</td>
      <td>289</td>
      <td>Evita (1996)</td>
    </tr>
    <tr>
      <th>81824</th>
      <td>302</td>
      <td>271</td>
      <td>Starship Troopers (1997)</td>
    </tr>
    <tr>
      <th>84234</th>
      <td>302</td>
      <td>333</td>
      <td>Game, The (1997)</td>
    </tr>
  </tbody>
</table>
</div>



## Step 4 - Optional Clean up <a id="cleanup"></a>


```python
# optionally uncomment and run the code to clean everything up

#!rm ml-100k.zip 2> /dev/null
#!rm recommendation.requests.out 2> /dev/null
#!rm -fr ml-100k/ 2> /dev/null
#!rm -fr {TRAIN_DATA_DIR} 2> /dev/null
#!rm -fr {BATCH_INPUT_DIR} 2> /dev/null
#sagemaker.delete_model(ModelName= model_name)
```
