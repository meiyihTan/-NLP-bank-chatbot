import pandas as pd
from termcolor import colored

message_answer_dict = pd.read_excel(
    r"Database.xlsx", sheet_name='COMMON_MESSAGES', index_col=0).to_dict(orient='index')


def get_common_answer(message):
    for key in message_answer_dict:
        if message.lower() in key:
            return message_answer_dict[key]['response']
    return None


if __name__ == '__main__':
    def display_chatbot_reply(answer):
        """
        Print the chatbot reply message in color 
        """
        reply = "Chatbot: "
        reply += colored(answer, "green")
        print(reply)

    print(message_answer_dict)
    print(get_common_answer('Hi'))
    print(get_common_answer('Hello'))
    print(get_common_answer('Goodbye'))
    while True:
        user_message = input()
        if user_message == "q":
            break

        # Return fixed responses for common messages like: "Hi"
        common_answer = get_common_answer(user_message)
        if common_answer:
            display_chatbot_reply(get_common_answer(user_message))
            continue

        print('Do something else')
