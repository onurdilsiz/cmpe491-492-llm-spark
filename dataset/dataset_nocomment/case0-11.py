from pyspark import SparkContext

sc = SparkContext("local", "Movie Ratings Analysis")
raw_ratings = sc.textFile("ratings.csv")

parsed_ratings = raw_ratings.map(lambda line: line.split(","))

high_ratings = parsed_ratings.filter(lambda x: float(x[2]) >= 3)

movie_counts = high_ratings.map(lambda x: (x[1], 1))

movie_rating_counts = movie_counts.reduceByKey(lambda x, y: x + y)

movie_count_key = movie_rating_counts.map(lambda x: (x[1], x[0]))

sorted_movies = movie_count_key.sortByKey(ascending=False)

top_10_movies = sorted_movies.take(10)
print("Top 10 most-rated movies:")
for count, movie_id in top_10_movies:
    print(f"Movie ID: {movie_id}, Ratings Count: {count}")

movie_ratings = parsed_ratings.map(lambda x: (x[1], (float(x[2]), 1)))

movie_rating_totals = movie_ratings.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

movie_average_ratings = movie_rating_totals.map(lambda x: (x[0], x[1][0] / x[1][1]))

movie_rating_data = movie_rating_counts.join(movie_average_ratings)

popular_movies = movie_rating_data.filter(lambda x: x[1][0] >= 50)

final_movies = popular_movies.map(lambda x: (x[0], x[1]))

print("Popular movies with more than 50 ratings:")
for movie_id, (count, avg_rating) in final_movies.collect():
    print(f"Movie ID: {movie_id}, Ratings Count: {count}, Average Rating: {avg_rating}")

distinct_users = parsed_ratings.map(lambda x: x[0]).distinct().count()
print(f"Total number of distinct users: {distinct_users}")

highest_rated_movie = popular_movies.sortBy(lambda x: -x[1][1]).take(1)
if highest_rated_movie:
    movie_id, (count, avg_rating) = highest_rated_movie[0]
    print(f"Highest-rated movie: Movie ID: {movie_id}, Ratings Count: {count}, Average Rating: {avg_rating}")

final_movies.saveAsTextFile("popular_movies_output")

sc.stop()
