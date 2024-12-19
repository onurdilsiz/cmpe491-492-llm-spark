# Spark Context Initialization
from pyspark import SparkContext

sc = SparkContext("local", "Movie Ratings Analysis")

# Load the raw data from a text file
# Assume the data format is: user_id,movie_id,rating,timestamp
raw_ratings = sc.textFile("ratings.csv")

# Split each line into a list of values
parsed_ratings = raw_ratings.map(lambda line: line.split(","))

# Filter out rows where the rating is below 3
high_ratings = parsed_ratings.filter(lambda x: float(x[2]) >= 3)

# Map the data to key-value pairs of (movie_id, 1) for counting occurrences
movie_counts = high_ratings.map(lambda x: (x[1], 1))

# Reduce by key to count the number of ratings for each movie
movie_rating_counts = movie_counts.reduceByKey(lambda x, y: x + y)

# Map the movie rating counts to format (count, movie_id) for sorting
movie_count_key = movie_rating_counts.map(lambda x: (x[1], x[0]))

# Sort movies by the number of ratings in descending order
sorted_movies = movie_count_key.sortByKey(ascending=False)

# Collect and print the top 10 most-rated movies
top_10_movies = sorted_movies.take(10)
print("Top 10 most-rated movies:")
for count, movie_id in top_10_movies:
    print(f"Movie ID: {movie_id}, Ratings Count: {count}")

# Calculate the average rating for each movie
# Map to (movie_id, (rating, 1)) for aggregation
movie_ratings = parsed_ratings.map(lambda x: (x[1], (float(x[2]), 1)))

# Aggregate by key to calculate the total rating and count per movie
movie_rating_totals = movie_ratings.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

# Map to calculate the average rating (movie_id, average_rating)
movie_average_ratings = movie_rating_totals.map(lambda x: (x[0], x[1][0] / x[1][1]))

# Join the average ratings with the rating counts
movie_rating_data = movie_rating_counts.join(movie_average_ratings)

# Filter movies with fewer than 50 ratings
popular_movies = movie_rating_data.filter(lambda x: x[1][0] >= 50)

# Map the final results to format (movie_id, (ratings_count, average_rating))
final_movies = popular_movies.map(lambda x: (x[0], x[1]))

# Collect and display the results
print("Popular movies with more than 50 ratings:")
for movie_id, (count, avg_rating) in final_movies.collect():
    print(f"Movie ID: {movie_id}, Ratings Count: {count}, Average Rating: {avg_rating}")

# Perform additional transformations for custom analysis
# Calculate the total number of users who rated movies
distinct_users = parsed_ratings.map(lambda x: x[0]).distinct().count()
print(f"Total number of distinct users: {distinct_users}")

# Find the movie with the highest average rating among popular movies
highest_rated_movie = popular_movies.sortBy(lambda x: -x[1][1]).take(1)
if highest_rated_movie:
    movie_id, (count, avg_rating) = highest_rated_movie[0]
    print(f"Highest-rated movie: Movie ID: {movie_id}, Ratings Count: {count}, Average Rating: {avg_rating}")

# Save the final results to a text file
final_movies.saveAsTextFile("popular_movies_output")

# Stop the Spark context
sc.stop()
