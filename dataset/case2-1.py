from pyspark.sql import SparkSession
import sys

#Define input path
input_file = sys.argv[1] #('numbers.txt')
print ("input path: ",input_file)


# In[ ]:


# Create an instance of a SparkSession object
spark = SparkSession.builder.appName("Quiz 3").getOrCreate()


# In[ ]:


#Create a new RDD[String]
file_q3 = spark.sparkContext.textFile(input_file)
file_q3.collect()


# In[ ]:


#Apply transformations
flat_q3 = file_q3.flatMap(lambda x: x.split())
flat_q3.collect()


# In[ ]:


#Define function for output
def is_number(iterator):
    C = 0 # the total count of all numbers
    Z = 0 # the total number of zeros
    P = 0 # the total number of positive numbers
    N = 0 # the total number of negative numbers
    S = 0 # the total number of non-numbers dropped
    
    for x in iterator:
        if ((x.strip('-')).isnumeric() == True):
            C = C + 1
            int_x = int(x)
            if (int_x == 0):
                Z = Z + 1
            if (int_x > 0):
                P = P + 1
            if (int_x < 0):
                N = N + 1
        else:
            S = S + 1
            
    return [(C, Z, P, N, S)]


# In[ ]:


#Ouput
map_q3 = flat_q3.mapPartitions(is_number)
finalrdd = map_q3.reduce(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3],x[4]+y[4]))

print(finalrdd)