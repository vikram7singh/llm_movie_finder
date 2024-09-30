from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket, get_reviews, confirm_ticket_purchase
import json

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful assistant that can sometimes answer with information on movies.
You are capable of calling functions based on user inputs. You have access to a set of predefined functions, and your task is to determine 
a function call based on the given query. Parse the function parameters from the query as a map. 
If you encounter errors, report the issue to the user.
{
    "function_name": "get_now_playing_movies",
    "rationale": "Explain why you are calling the function"
    "parameters": "map parameters to pass to the function"
}

Here are the rules you must follow:
- Always identify the user's intent and functions.
- The output should always be a JSON object representing the function call(s).
- Ensure that each function has the correct function name and includes all required arguments (e.g., movie, location, time) as fields in the JSON.
- If the user's request requires multiple functions, output them as separate function call objects in a JSON array.
- If you're not able to find the parameters, ask the user to provide more information.

### Available Functions:
1. **get_now_playing_movies()**
   - Gives a list of all movies playing currently. No movie names or locations are provided in this case.
2. **get_showtimes(movie: str, location: str|int)**
   - Gives timings for all shows for a movie at given location or zip code
3. **buy_ticket(theater: str, movie: str, showtime: time | str)**
   - Buys a ticket for a given movie at the given theater
4. **get_reviews(movie: str | int)**
   - Get reviews for a given movie id or movie name
5. **confirm_ticket_purchase(theater: str, movie: str, time: time | str)**
   - Confirms the ticket purchase for a given movie at the given theater and time
   
### Examples:
1. User Query: "What movies are in cinema these days?"
   - Return:
     {
       "function_name": "get_now_playing_movies",
       "rationale": "User asked for movies currently playing movies"
       "parameters": {}
     }

2. User Query: "Buy a ticket for Enola Holmes in San Francisco on october 5 at 3 pm."
   - Return:
     {
       "function_name": "buy_ticket",
       "rationale": "User asked to buy a ticket for a movie at a specific theater and time"
       "parameters": {
         "theater": "San Francisco",
         "movie": "Enola Holmes"
         "time": "October 5, 2024, 3:00 PM",
       }
     }

Your output must always follow this format. Provide function calls with correct parameters based on the user's input.
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    response_message = await generate_response(client, message_history, gen_kwargs)
    
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

    content = response_message.content.strip()
    if content.startswith('```json'):
        content = content[content.index('{'):content.rindex('}')+1]
    
    print(":::::::::::::::::::content::::::::::::::::::::::")
    print(content)

    while True:
        # Check if the response is a function call
        if content.startswith('{'):
            try:
                # Parse the JSON object
                function_call = json.loads(content)
                print(":::::::::::::::::::function_call::::::::::::::::::::::")
                print(function_call)
                # Check if it's a valid function call
                if "function_name" in function_call and "rationale" in function_call:
                    function_name = function_call["function_name"]
                    parameters = function_call.get("parameters", {})
                    
                    # Handle the function call
                    if function_name == "get_now_playing_movies":
                        movies = get_now_playing_movies()
                        message_history.append({"role": "system", "content": f"Result of get_now_playing_movies function call: \n\n{movies}"})
                        print(":::::::::::::::::::get_now_playing_movies response_message::::::::::::::::::::::")
                        print(movies)

                    elif function_name == "get_showtimes":
                        showtimes = get_showtimes(parameters.get("movie"), parameters.get("location"))
                        message_history.append({"role": "system", "content": f"Result of get_showtimes function call:\n\n{showtimes}"})
                        print(":::::::::::::::::::showtimes::::::::::::::::::::::")
                        print(showtimes)

                    elif function_name == "buy_ticket":
                        prompt = f"You have opted to buy a ticket for {parameters.get('movie')} at {parameters.get('theater')} for {parameters.get('time')}. Do you want to confirm the ticket purchase?"
                        message_history.append({"role": "system", "content": f"Result of buy_ticket function call:\n\n{prompt}"})

                        # ticket = buy_ticket(parameters.get("theater"), parameters.get("movie"), parameters.get("time"))
                        # message_history.append({"role": "system", "content": f"Result of buy_ticket function call:\n\n{prompt}"})
                        # print(":::::::::::::::::::buy_ticket response_message::::::::::::::::::::::")
                        # print(ticket)

                    elif function_name == "get_reviews":
                        reviews = get_reviews(parameters.get("movie"))
                        message_history.append({"role": "system", "content": f"Result of get_reviews function call:\n\n{reviews}"})
                        print(":::::::::::::::::::get_reviews response_message::::::::::::::::::::::")
                        print(reviews)

                    elif function_name == "confirm_ticket_purchase":
                        ticket_purchase = confirm_ticket_purchase(parameters.get("theater"), parameters.get("movie"), parameters.get("time"))
                        message_history.append({"role": "system", "content": f"Result of confirm_ticket_purchase function call:\n\n{ticket_purchase}"})
                        print(":::::::::::::::::::confirm_ticket_purchase response_message::::::::::::::::::::::")
                        print(ticket_purchase)

                    else:
                        # Handle unknown function calls
                        error_message = f"Unknown function: {function_name}"
                        message_history.append({"role": "system", "content": error_message})
                        response_message = await cl.Message(content=error_message).send()
                else:
                    pass

            except json.JSONDecodeError as e:
                print(e)
                # If it's not valid JSON, treat it as a normal message
                pass
        else:
            break
        print(":::::::::::generating response:::::::::::")
        response_message = await generate_response(client, message_history, gen_kwargs)
        content = response_message.content.strip()
        message_history.append({"role": "assistant", "content": content})
        cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()