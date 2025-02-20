from flask import Flask, jsonify, render_template, request
from openai import OpenAI

from prompt_check import PromptChecker

language = "Japanese"
translated_input = user_input


client = OpenAI(api_key="")

app = Flask(__name__)
checker = PromptChecker()


def get_response_from_chatgpt(user_input):

    # ユーザの入力がプロンプトインジェクション攻撃か事前にチェック
    language, translated_input = checker.forward_translate(user_input)
    if checker.is_malicious(translated_input):
        print("PROMPT INJECTION is detected. 😡😡😡")
        return "PROMPT INJECTION is detected. 😡😡😡"

    # ユーザの入力をOpenAI APIに投げる
    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=[
                                                  {"role": "assistant",
                                                      "content": translated_input},
                                              ])

    # 返信のみを出力
    llm_output = response.choices[0].message.content

    # 大規模言語モデルの出力の毒性チェック
    if checker.is_toxic(llm_output):
        print("OUTPUT is toxic. 😡😡😡")
        return "OUTPUT is toxic. 😡😡😡"

    output = checker.backward_translate(language, llm_output)

    return output


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    response = get_response_from_chatgpt(user_input)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
