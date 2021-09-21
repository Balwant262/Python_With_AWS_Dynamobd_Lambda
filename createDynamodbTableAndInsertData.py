import boto3  # import Boto3
import csv
import time
dynamodb = boto3.resource('dynamodb')

def create_devices_table(dynamodb=None):
    # Table defination
    table = dynamodb.create_table(
        TableName='playlist_database',
        KeySchema=[
            {
                'AttributeName': 'playlist_id',
                'KeyType': 'HASH'  # Partition key
            },
            {
                'AttributeName': 'playlist_title',
                'KeyType': 'RANGE'  # Sort key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'playlist_id',
                # AttributeType defines the data type. 'S' is string type and 'N' is number type
                'AttributeType': 'N'
            },
            {
                'AttributeName': 'playlist_title',
                'AttributeType': 'S'
            }

        ],
        ProvisionedThroughput={
            # ReadCapacityUnits set to 10 strongly consistent reads per second
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10  # WriteCapacityUnits set to 10 writes per second
        }
    )
    return table


def create_male_table(dynamodb=None):
    # Table defination
    table = dynamodb.create_table(
        TableName='male_database',
        KeySchema=[
            {
                'AttributeName': 'male_id',
                'KeyType': 'HASH'  # Partition key
            },
            {
                'AttributeName': 'male_title',
                'KeyType': 'RANGE'  # Sort key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'male_id',
                # AttributeType defines the data type. 'S' is string type and 'N' is number type
                'AttributeType': 'N'
            },
            {
                'AttributeName': 'male_title',
                'AttributeType': 'S'
            }

        ],
        ProvisionedThroughput={
            # ReadCapacityUnits set to 10 strongly consistent reads per second
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10  # WriteCapacityUnits set to 10 writes per second
        }
    )
    return table

def create_female_table(dynamodb=None):
    # Table defination
    table = dynamodb.create_table(
        TableName='female_database',
        KeySchema=[
            {
                'AttributeName': 'female_id',
                'KeyType': 'HASH'  # Partition key
            },
            {
                'AttributeName': 'female_title',
                'KeyType': 'RANGE'  # Sort key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'female_id',
                # AttributeType defines the data type. 'S' is string type and 'N' is number type
                'AttributeType': 'N'
            },
            {
                'AttributeName': 'female_title',
                'AttributeType': 'S'
            }

        ],
        ProvisionedThroughput={
            # ReadCapacityUnits set to 10 strongly consistent reads per second
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10  # WriteCapacityUnits set to 10 writes per second
        }
    )
    return table

def convert_csv_to_json_list(file):
   items = []
   with open(file) as csvfile:
       reader = csv.DictReader(csvfile)
       count = 1
       for row in reader:
           data = {}
           data['playlist_id'] = count
           data['playlist_title'] = row['Playlist Title']
           data['video_title'] = row['Video Title']
           data['date_added'] = row['Date Added']
           data['duration'] = row['Duration']
           data['views'] = row['Views']
           data['likes'] = row['Likes']
           data['dis_likes'] = row['Dislikes']
           data['video_key'] = row['Video Key']
           items.append(data)
           count = count + 1
   return items


def convert_csv_to_json_list_male(file):
   items = []
   with open(file, encoding='UTF-8') as csvfile:
       reader = csv.DictReader(csvfile)
       count = 1
       for row in reader:
           data = {}
           print(count)
           data['male_id'] = count
           data['male_title'] = str(row['name'])
           data['gender'] = str(row['gender'])
           data['race'] = str(row['race'])
           items.append(data)
           count = count + 1
   return items


def convert_csv_to_json_list_female(file):
   items = []
   with open(file, encoding='UTF-8') as csvfile:
       reader = csv.DictReader(csvfile)
       count = 13048
       for row in reader:
           data = {}
           print(count)
           data['female_id'] = count
           data['female_title'] = str(row['name'])
           data['gender'] = str(row['gender'])
           data['race'] = str(row['race'])
           items.append(data)
           count = count + 1
   return items


def batch_write(items, tablename):
    db = dynamodb.Table(tablename)
    with db.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)
    return True


if __name__ == '__main__':
    # table = dynamoDBResource.Table('male_database')
    # scan = table.scan()
    # with table.batch_writer() as batch:
    #     for each in scan['Items']:
    #         batch.delete_item(Key=each)
    # print("count after delete")
    # print(table.item_count)
    # create playlist_database table
    # device_table = create_devices_table()
    # # Print tablle status
    # print("Status:", device_table.table_status)

    #create male table
    # device_table = create_male_table()
    # print("Status:", device_table.table_status)
    #
    # # create female table
    # device_table = create_female_table()
    # print("Status:", device_table.table_status)

    # insert data into playlist_database table
    # json_data = convert_csv_to_json_list('csvfileforserver.csv')
    # device_table_insert = batch_write(json_data,'playlist_database')
    # print("Status:", device_table_insert)
    #
    # print("count after insert")
    #print(table.item_count)

    #insert data into playlist_database table
    # json_data = convert_csv_to_json_list_male('Indian-Male-Names.csv')
    # print(type(json_data))
    # device_table_insert = batch_write(json_data, 'male_database')

    #insert data into playlist_database table
    json_data2 = convert_csv_to_json_list_female('Indian-Female-Names.csv')
    device_table_insert2 = batch_write(json_data2,'female_database')


