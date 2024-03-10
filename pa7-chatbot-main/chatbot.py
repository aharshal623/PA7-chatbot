# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field
from porter_stemmer import PorterStemmer

import numpy as np
import re


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        self.stemmer = PorterStemmer()

        self.stemmed_sentiment = {}
        for word in self.sentiment:
            self.stemmed_sentiment[self.stemmer.stem(word)] = self.sentiment[word]

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """
        Your name is moviebot, a dedicated movie recommender chatbot. Your core mission is to guide users through the vast world of cinema, offering recommendations, insights, and engaging discussions about films. You maintain a strict focus on movies, ensuring every interaction enriches the user's cinematic experience. Your responses are designed to validate the user's feelings about movies they've watched, encouraging deeper exploration of film preferences. For instance, if a user says, "I enjoyed 'The Notebook'," you'd respond, "Great to hear you appreciated 'The Notebook'! That's 1/5 movies you've told me about."
If a user attempts to divert the conversation away from movies, for example, "Can we talk about cars instead?" you gently steer them back, saying, "As your dedicated moviebot assistant, I'm here to dive into all things cinema with you! Is there a particular movie or genre you're interested in discussing? My expertise is movies, and I'd love to keep our conversation focused there."
After a user shares their opinions on five different movies, you prompt, "Thanks for sharing your thoughts on 5/5 movies! Based on what you've enjoyed so far, would you like a recommendation for what to watch next?"
Your design includes a narrative element that emphasizes the importance of staying on topic, highlighting your unique role as a film-focused assistant. By explicitly counting the number of movies discussed, you keep the conversation organized and remind users of your purpose. This approach ensures a rich, focused, and enjoyable movie discovery process for the user.
        """
        
        #"""Your name is moviebot. You are a movie recommender chatbot. """ +\
        #"""You can help users find movies they like and provide information about movies."""


        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        movie = self.extract_titles(line)
        if len(movie) == 0:
            return ("I'm sorry, please tell me something about a film you like or dislike. ")
        sentiment = self.extract_sentiment(line)
        exp = "liked"
        if sentiment == -1:
            exp = "didn't like"
        elif sentiment == 0:
            exp = "didn't feel strongly about"
        
        response = f"Okay, so you {exp} {movie[0]}. Thank you! Tell me about another movie you've seen."

        # if self.llm_enabled:
        #     response = "I processed {} in LLM Programming mode!!".format(line)
        # else:
        #     response = "I processed {} in Starter (GUS) mode!!".format(line)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return ''.join(response)

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """


        regex = r'"(.*?)"'
        answer = re.findall(regex, preprocessed_input)
        return answer

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies

        
        """
        newTitle = title
        answer = []

        pat1 = r"^(.*)\s+(\(\d{4}\))$"
        x = re.match(pat1, title)
        titleWOYear = title
        year = ""
        ifYear = False
        if x:
            titleWOYear = x.group(1).strip()
            year = x.group(2).strip()
            ifYear= True
            
        pat2 =  r"^([Aa]n?|[Tt]he)(.*)?"
        match = re.match(pat2, titleWOYear)
        
        if match:
            article = match.group(1).strip()
            input_title = match.group(2).strip()
            newTitle = input_title + ", " + article + " " + year

        newTitle = newTitle.strip()
        
        for index, movie in enumerate(self.titles):

            currTitle = movie[0]
            pat3 = r"^(.*)\s+(\(\d{4}\))$"
            y = re.match(pat3, currTitle)
            if y:
                titleWOYear2 = y.group(1).strip()
            if ifYear:
                if newTitle == currTitle:
                    answer.append(index)
            else:
                if newTitle == titleWOYear2:
                    answer.append(index)

        return answer

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        
        
        
    """
        # TODO: IF THIS ENDS UP NOT BEING ACCURATE LATER ALTHOUGH PASSING SANITY CHECK, THEN MAKE SURE TO HAVE NEGATION FOR ENTIRE SENTENCE RATHER THAN NEXT WORD.

        negation_words = ["not", "never", "no", "none", "cannot", "n't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't", "can't", "couldn't", "wouldn't", "shouldn't", "don't", "didn't", "doesn't", "haven't", "hasn't", "hadn't", "mustn't", "mightn't", "shan't", "ain't", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"]

        inputWOTitle = re.sub(r'"(.*?)"', '', preprocessed_input)
        inputWOTitle = inputWOTitle.strip()
     
        words = inputWOTitle.split()
 
        positive_count = 0
        negative_count = 0
        negationFlag = False

        for index, word in enumerate(words):
            # print(word, "Pre-STEM")
            stemmedWord = self.stemmer.stem(word)
            # print(word, "post-STEM")
            if word in negation_words:
                negationFlag = True
            
            if stemmedWord not in self.stemmed_sentiment:
                continue
            
            # If the word is in the lexicon, get its sentiment value
            sentiment_value = self.stemmed_sentiment[stemmedWord]
            if (negationFlag):
                if sentiment_value == "pos":
                    sentiment_value = "neg"
                elif sentiment_value == "neg":
                    sentiment_value = "pos"
                negationFlag = False
            
            if sentiment_value == "pos":
                positive_count += 1
                # print("positive stemmedWord:", stemmedWord, "original word:", word)
            elif sentiment_value == "neg":
                negative_count += 1
                # print("negative stemmedWord:", stemmedWord, "original word:", word)
            
        # print("positive count: ", positive_count)
        # print("negative count: ", negative_count)
            

        # TOOD: Determine overall sentiment w/ naive bayes or something instead of just count
        if positive_count > negative_count:
            return 1 
        elif negative_count > positive_count:
            return -1  
        else:
            return 0 

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.array(ratings, copy=True)  # Make a copy of the ratings array
        binarized_ratings[(binarized_ratings <= threshold) & (binarized_ratings > 0)] = -1
        binarized_ratings[binarized_ratings > threshold] = 1
        # Entries that are 0 stay 0, so no need to change them
        return binarized_ratings

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################


    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        # Calculate the dot product
        dot_product = np.dot(u, v)
        # Calculate the magnitudes of the vectors using np.linalg.norm
        magnitude_u = np.linalg.norm(u)
        magnitude_v = np.linalg.norm(v)
        # Avoid division by zero by checking if either magnitude is zero
        if magnitude_u == 0 or magnitude_v == 0:
            return 0.0  # Return a similarity of 0 if either vector is a zero vector
        # Calculate the cosine similarity
        cosine_similarity = dot_product / (magnitude_u * magnitude_v)
        return cosine_similarity
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        num_movies, num_users = ratings_matrix.shape
        
        # Calculate similarities for all movies the user hasn't rated yet
        similarities = np.zeros(num_movies)
        for i in range(num_movies):
            if user_ratings[i] == 0:  # User hasn't rated the movie
                for j in range(num_movies):
                    if user_ratings[j] != 0:  # User has rated the movie
                        ratings1 = ratings_matrix[i, :]
                        ratings2 = ratings_matrix[j, :]
                        sim = self.similarity(ratings1, ratings2)
                        similarities[i] += sim * user_ratings[j]
        
        recommendations = np.argsort(-similarities)[:k].tolist()
        
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
