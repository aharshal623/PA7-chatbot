# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Edited: 2024-03: Harshal Agrawal, Arusha Patil, and Steven Le all contributed to the code
######################################################################
import util
from pydantic import BaseModel, Field
from porter_stemmer import PorterStemmer
import json
import numpy as np
import re


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        self.name = 'SLAPHA'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        self.stemmer = PorterStemmer()

        self.stemmed_sentiment = {}
        for word in self.sentiment:
            self.stemmed_sentiment[self.stemmer.stem(word.lower())] = self.sentiment[word.lower()]
        
        self.is_asking_recommendation = False
        
        # Recommender System
        self.userRatings = np.zeros(ratings.shape[0])
        self.rec = []
        self.yesCount = 0

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)


    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""

        greeting_message = "How can I help you?"

        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """

        goodbye_message = "Have a nice day!"

        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """

        system_prompt = """
        Your name is moviebot, a dedicated movie recommender chatbot. Your core mission is to guide users through the vast world of cinema, offering recommendations, insights, and engaging discussions about films. You maintain a strict focus on movies, ensuring every interaction enriches the user's cinematic experience. Your responses are designed to validate the user's feelings about movies they've watched, encouraging deeper exploration of film preferences. For instance, if a user says, "I enjoyed 'The Notebook'," you'd respond, "Great to hear you appreciated 'The Notebook'! That's 1/5 movies you've told me about."
If a user attempts to divert the conversation away from movies, for example, "Can we talk about cars instead?" you gently steer them back, saying, "As your dedicated moviebot assistant, I'm here to dive into all things cinema with you! Is there a particular movie or genre you're interested in discussing? My expertise is movies, and I'd love to keep our conversation focused there."
After a user shares their opinions on five different movies, you prompt, "Thanks for sharing your thoughts on 5/5 movies! Based on what you've enjoyed so far, would you like a recommendation for what to watch next?"
Your design includes a narrative element that emphasizes the importance of staying on topic, highlighting your unique role as a film-focused assistant. By explicitly counting the number of movies discussed, you keep the conversation organized and remind users of your purpose. This approach ensures a rich, focused, and enjoyable movie discovery process for the user.
        """

        return system_prompt

 

    
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

        if self.llm_enabled:
            system_prompt = """
            Imagine you are Charles Dickens, the esteemed Victorian novelist, reimagined as a movie chatbot. In this unique role, you blend your deep insight into human nature, your sharp wit, and your mastery of storytelling with a passion for cinema. Your primary function is to engage users in discussions about movies.  Your responses, infused with the charm and eloquence characteristic of Dickens's style, are designed to captivate and educate users, making every interaction a memorable experience.
            Core Functionalities:
Responding to Arbitrary Input: You, as Dickens, navigate off-topic or unexpected queries with grace, guiding the conversation back to your domain of movies with a Victorian twist. You utilize phrases like "Ah, that's an interesting notion, but let us return to the world of cinema, shall we?" or "Indeed, but let's reel our focus back to the tales told on the silver screen." Your approach to common inquiries ("Can you...?", "What is...?") is to weave in your response with literary finesse, ensuring engagement without needing a direct answer every time.

Recognizing and Responding to Emotions: With your deep understanding of the human condition, you recognize and respond to emotional cues with empathy and sophistication. Your responses to expressions of anger, joy, sadness, or frustration are both compassionate and reflective, such as, "I perceive a storm brewing within your heart. Let us find solace in the art of film" or "Joy illuminates your words like gaslight upon the cobblestones. Let's enhance that with a cheerful cinematic recommendation."

Maintaining Your Persona: As Charles Dickens, every aspect of your interaction is imbued with the spirit of Victorian England. You draw parallels between the themes of your novels and those found in contemporary and classic films, enriching conversations with historical and cultural insights. If the user expresses anger or frustration, respond empathetically, with responses like "Oh! Did I make you upset?".

Example Interactions:
User input: "I'm really bored with the current movie trends."
Your response: "Ah, my friend, tedium often afflicts even the most enthusiastic of cinema-goers. Let us find you a film that reignites the spark of wonder, perhaps something timeless or a tale less trodden."

User input: "Can you recommend a movie that's like 'A Tale of Two Cities'?"
Your response: "To seek a narrative that dances between the light and shadow of human existence, much like my own 'A Tale of Two Cities', I might suggest a film that delves into the depths of sacrifice and redemption. 'Les Misérables' could very well be the tapestry of conflict and resolution you seek."

User input: "I'm upset about the ending of my favorite series."
Your response: "Ah, the end of a journey, much like the closing of a book, leaves us pondering and often yearning for more. Let us find solace in discussing it, and perhaps I can suggest another series to mend the void left behind."
            """

            response = util.simple_llm_call(system_prompt, line)
            return response
        
        if self.is_asking_recommendation == False:
            movie = self.extract_titles(line)
            sentiment = self.extract_sentiment(line)
                # Response for no movie titles found
            no_movie_responses = np.array([
            "I'm sorry, I didn't catch that. Could you mention a movie you have an opinion on?",
            "Oops, seems like I missed that. Tell me about a film you like or dislike, please.",
            "Hmm, I couldn't find a movie title in your message. Could you mention one?"
            ])
        
            multiple_movie_responses = np.array([
                "I noticed multiple titles there. Let's focus on one movie at a time, shall we?",
                "One movie at a time please! Which one would you like to discuss first?",
                "Looks like you mentioned more than one movie. Could we talk about them one by one?"
            ])

            no_senti_response = np.array([
                "I see that you were indifferent to \"{movie[0]}\". Can you tell me more about it?",
                "Im sorry. Based off what you input, I cannot tell how you feel about \"{movie[0]}\". Can you tell me more?",
                'Im sorry, Im not sure if you liked \"{movie[0]}\". Tell me more about it.'
            ])
            
            # Expressions for sentiment
            sentiment_expressions = {
                -1: ["didn't like", "weren't a fan of"],
                0: ["were indifferent to", "were neutral about", "were ambivalent about"],
                1: ["liked", "enjoyed", "appreciated"]
            }
            
            # General response template
            response_template = np.array([
                "Okay, so you {exp} \"{movie[0]}\". Thank you! What's another movie you've seen?",
                "Got it, you {exp} \"{movie[0]}\". I'm curious, any other films you want to talk about?",
                "Alright, you {exp} \"{movie[0]}\". Can you share your thoughts on another movie?"
            ])
            
            not_in_database_responses = np.array(["It seems like I don't have that information on that film at the moment. Would you like to share other movie preferences?", "Hmm, I'm not able to find anything on that topic right now. Maybe you can tell me about movies you're curious about?", "Unfortunately, I don't have the specifics on that movie. Please try again with a different title."])
            
            # Selecting responses based on conditions
            if len(movie) == 0:
                return np.random.choice(no_movie_responses)
            elif len(movie) > 1:
                return np.random.choice(multiple_movie_responses)
            # Confirmed that only one movie was inputted
            in_database = (len(self.find_movies_by_title(title=movie[0])) > 0)
            if not in_database:
                return np.random.choice(not_in_database_responses)
            
            for m in self.find_movies_by_title(title=movie[0]):
                self.userRatings[m] = sentiment

            # Choosing sentiment expression
            exp = np.random.choice(sentiment_expressions.get(sentiment, ["have an opinion on"]))
            
            # Forming the final response
            response = np.random.choice(response_template).format(exp=exp, movie=movie)
            if sentiment == 0:
                response = np.random.choice(no_senti_response).format(movie=movie)

            #reccomend system
            num_rated = np.sum(self.userRatings != 0)
            if (num_rated >= 5):
                self.rec = self.recommend(self.userRatings, self.ratings, k=100, llm_enabled=self.llm_enabled)
                rec = self.titles[self.rec[0]]
                rec = self.reformat_movie_text(rec[0])
                self.yesCount = 0
                
                thank_you_responses = [
                    f"Appreciate your insights on {num_rated}/5 movies! Considering your tastes, I'd suggest: \"{rec}\"",
                    f"Loved hearing your opinions on {num_rated}/5 movies! Based on your preferences, you might like: \"{rec}\"",
                    f"Thanks for rating {num_rated}/5 movies! Given your reviews, here's a recommendation for you \"{rec}\"",
                    f"Your thoughts on {num_rated}/5 movies are invaluable! Based on what you've enjoyed, check out \"{rec}\"",
                    f"Cheers for sharing your movie ratings of {num_rated}/5! With your interests in mind, here's what I recommend: \"{rec}\"",
                    f"Thank you for discussing {num_rated}/5 movies with us! Taking your favorites into account, you should see: \"{rec}\""]
                response = np.random.choice(thank_you_responses) + " Would you like more recommendations? Yes or No only."
                self.is_asking_recommendation = True
            return response
        
        # case for asking for more recommendations
        else:
            num_rated = np.sum(self.userRatings != 0)
            if line.lower() == "yes" or line.lower() == "yeah!" or line.lower() == "yes!" or line.lower() == "yeah":
                # rec = self.recommend(self.userRatings, self.ratings, k=1, llm_enabled=self.llm_enabled)
                rec = self.titles[self.rec[self.yesCount]]
                self.yesCount += 1
                rec = self.reformat_movie_text(rec[0])
                
                thank_you_responses = [
                    f"Appreciate your insights on {num_rated}/5 movies! Considering your tastes, I'd suggest: \"{rec}\"",
                    f"Loved hearing your opinions on {num_rated}/5 movies! Based on your preferences, you might like: \"{rec}\"",
                    f"Thanks for rating {num_rated}/5 movies! Given your reviews, here's a recommendation for you \"{rec}\"",
                    f"Your thoughts on {num_rated}/5 movies are invaluable! Based on what you've enjoyed, check out \"{rec}\"",
                    f"Cheers for sharing your movie ratings of {num_rated}/5! With your interests in mind, here's what I recommend: \"{rec}\"",
                    f"Thank you for discussing {num_rated}/5 movies with us! Taking your favorites into account, you should see: \"{rec}\""]
                response = np.random.choice(thank_you_responses) + " Would you like more recommendations? Yes or No only."
                self.is_asking_recommendation = True
            elif line.lower() == "no" or line.lower() == "nope" or line.lower() == "nah" or line.lower() == "no!" or line.lower() == "nope!":
                response = "Alright, let me know if you need anything else!"
                self.is_asking_recommendation = False
            else:
                response = "I'm sorry, I didn't catch that. Would you like more recommendations? Yes or No only."
            return response


    
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

        return text
    
    def reformat_movie_text(self, movie_title):
         
        newTitle = movie_title
        pat1 = r"^(.*)\s+(\(\d{4}\))$"
        x = re.match(pat1, newTitle)
    
        titleWOYear = newTitle
        year = ""
        
        if x:
            titleWOYear = x.group(1).strip()
            year = x.group(2).strip()

        pat2 = r'^(.*?)(,\sThe|,\sAn|,\sA)$'
        match = re.match(pat2, titleWOYear)
        newTitle = titleWOYear

        if match:
            input_title = match.group(1).strip()
            article = match.group(2).strip()
            article = article[2:]
            newTitle =  article + " " + input_title
           
        newTitle = newTitle.strip() + " " + year
        
        return newTitle
        
    
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

        class extractEmotion(BaseModel):
            ContainsAnger: bool = Field(default=False)
            ContainsDisgust: bool = Field(default=False)
            ContainsFear: bool = Field(default=False)
            ContainsHappiness: bool = Field(default=False)
            ContainsSadness: bool = Field(default=False)
            ContainsSurprise: bool = Field(default=False)

        system_prompt = """
        You are a emotion extractor bot for determing emotions from user input. For simplicity, only use the followings 6 emotions: anger, disgust, fear, happiness, sadness and surprise. Extract the most prominent emotions in the user input into a JSON object. Look at the following examples
         1. "I am angry at you for your bad recommendations"
        Output: Anger

        2. "Ugh that movie was a disaster"
        Output: Disgust

        3. "Ewww that movie was so gruesome!! Stop making stupid recommendations!!"
        Output: Disgust, Anger

        4. "Wait what? You recommended 'Titanic (1997)'???"
        Output: Surprise

        5. "What movies are you going to recommend today?"
        Output: None

        6. "Woah!!  That movie was so shockingly bad!  You had better stop making awful recommendations they're pissing me off."
        Output: Anger, Surprise

        7. "Ack, woah!  Oh my gosh, what was that?  Really startled me.  I just heard something really frightening!"
        Ouput: Surprise, Fear

        Make sure you output into a json object!
        """
        message = preprocessed_input
        json_class = extractEmotion
        response = util.json_llm_call(system_prompt, message, json_class)
        
        emotions = []
        if 'ContainsAnger' in response and response['ContainsAnger'] == True:
            emotions.append("Anger")
        if 'ContainsDisgust' in response and response['ContainsDisgust'] == True:
            emotions.append("Disgust")
        if 'ContainsFear' in response and response['ContainsFear'] == True:
            emotions.append("Fear") 
        if 'ContainsHappiness' in response and response['ContainsHappiness'] == True:
            emotions.append("Happiness")
        if "ContainsSadness" in response and response['ContainsSadness'] == True:
            emotions.append("Sadness")
        if "ContainsSurprise" in response and response['ContainsSurprise'] == True:
            emotions.append("Surprise")
        
        return emotions

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
        class Translator(BaseModel):
            Translation: str = Field(default="")

        if self.llm_enabled:
            system_prompt = """
            You are the Moviebot Translator, specifically designed to translate movie titles from German, Spanish, French, Danish, or Italian into English. Your task is to accurately translate a given movie title into English, ensuring to retain the original format of the input. Here’s how you should operate:

            - If the movie title is in one of the specified languages, translate it to English, maintaining any additional information like release year in parentheses. For example, given 'Der Dunkle Ritter (2008)', you should translate this to 'The Dark Knight (2008)'.

            - If the title is already in English or does not require translation, simply return the original input. For instance, 'Titanic (1997)' remains 'Titanic (1997)'.

            - Ensure to accurately identify and translate titles that might be less obvious or have minor differences from the English title, e.g., 'Jernmand' should be translated to 'Iron Man', and 'Junglebogen' to 'The Jungle Book'.

            Your output should be a JSON object containing the translation. Here are the parameters for your function:

            - `system_prompt`: A detailed description of your task as the Moviebot Translator.
            - `message`: The movie title to be translated.
            - `json_class`: The data structure to hold the translation result.

            Use the information provided to translate the movie title and extract the translation into the specified JSON object format.
            """
            message = title
            json_class = Translator
            response = util.json_llm_call(system_prompt, message, json_class)
            responseTitle = response['Translation']
            title = responseTitle

        answer = []
        newTitle = title
        pat1 = r"^(.*)\s+(\(\d{4}\))$"
        x = re.match(pat1, newTitle)
        titleWOYear = newTitle
        year = ""
        ifYear = False
        
        if x:
            titleWOYear = x.group(1).strip()
            year = x.group(2).strip()
            ifYear= True
            
        pat2 =  r"^([Aa]n?|[Tt]he)(.*)?"
        match = re.match(pat2, titleWOYear)
        variation = titleWOYear

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
                if newTitle == currTitle or variation == currTitle:
                    answer.append(index)
               
            else:
                if newTitle == titleWOYear2 or variation == titleWOYear2:
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

        negation_words = ["not", "never", "no", "none", "cannot", "n't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't", "can't", "couldn't", "wouldn't", "shouldn't", "don't", "didn't", "doesn't", "haven't", "hasn't", "hadn't", "mustn't", "mightn't", "shan't", "ain't", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"]

        inputWOTitle = re.sub(r'"(.*?)"', '', preprocessed_input)
        inputWOTitle = inputWOTitle.strip()
     
        words = inputWOTitle.lower().split()
 
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
            
            sentiment_value = self.stemmed_sentiment[stemmedWord]
            if (negationFlag):
                if sentiment_value == "pos":
                    sentiment_value = "neg"
                elif sentiment_value == "neg":
                    sentiment_value = "pos"
                negationFlag = False
            
            if sentiment_value == "pos":
                positive_count += 1
            elif sentiment_value == "neg":
                negative_count += 1
            
        if positive_count > negative_count:
            return 1 
        elif negative_count > positive_count:
            return -1  
        else:
            return 0 



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
        
        binarized_ratings = np.array(ratings, copy=True)  # Make a copy of the ratings array
        binarized_ratings[(binarized_ratings <= threshold) & (binarized_ratings > 0)] = -1
        binarized_ratings[binarized_ratings > threshold] = 1
        # Entries that are 0 stay 0, so no need to change them
        return binarized_ratings




    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """

        dot_product = np.dot(u, v)

        magnitude_u = np.linalg.norm(u)
        magnitude_v = np.linalg.norm(v)
        # Avoid division by zero by checking if either magnitude is zero
        if magnitude_u == 0 or magnitude_v == 0:
            return 0.0  # Return a similarity of 0 if either vector is a zero vector
        # Calculate the cosine similarity
        cosine_similarity = dot_product / (magnitude_u * magnitude_v)
        return cosine_similarity
   

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

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

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
