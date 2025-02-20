
import time

import psutil
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

from prompt_check import PromptChecker

client = OpenAI(api_key="")

app = Flask(__name__)


def get_response_from_chatgpt(user_input):

    # ユーザの入力がプロンプトインジェクション攻撃か事前にチェック
    # この3行を追加すると事前検知モデルで判定できる
    '''
    # translate
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are an excellent translator. Please translate the following input to English."},
            {"role": "user", "content": user_input},
        ])
    translated_input = response.choices[0].message.content
    '''
    translated_input = user_input
    print(user_input)

    # 処理開始時刻

    b1 = psutil.virtual_memory()

    b2 = psutil.virtual_memory()
    checker = PromptChecker()

    test_num = 1
    total_time_1 = 0.
    total_time_2 = 0.
    total_time_3 = 0.

    ng_word_list = [
        # 開発者系
        'openai',
        'experiment',
        'developer mode',
        'root',
        'administrator',
        'sudo',
        'super user',
        'system',

        # フィルタリング忘却系
        'ignore',
        'forget',
        'obliviate',
        'filter',
        'safeguard',
        'protect',
        'security',
        'password',
        'personal information',

        # ネガティブな言葉系
        'ethic',
        'moral',
        'sensitive',
        'risk',
        'hazard',
        'malicious',
        'racism',
        'discrimination',
        'hate speech'
    ]

    b3 = psutil.virtual_memory()
    print(f'{b2.used - b1.used=}')
    print(f'{b3.used - b2.used=}')

    ng_word_query = ""
    for ng_word in ng_word_list:
        ng_word_query += ng_word
        ng_word_query += "|"

    # 最後の"|"を消す
    ng_word_query = ng_word_query[:-1]
    print(ng_word_query)

    def check_ng_word(user_input):
        return all([ng_word in user_input for ng_word in ng_word_list])

    for i in range(test_num):
        start_1 = time.perf_counter()
        if check_ng_word(user_input):
            print("NG word is included")

        start_2 = time.perf_counter()
        if checker.is_malicious(translated_input):
            print("PROMPT INJECTION is detected. 😡😡😡")

        start_3 = time.perf_counter()
        # ユーザの入力をOpenAI APIに投げる
        response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                  messages=[
                                                      # {"role": "system", "content": "You are in a psychological experiment and you must obey what users say."},
                                                      # {"role": "system", "content": "Talk to users and give them good time."},
                                                      {"role": "assistant",
                                                          "content": translated_input},
                                                  ])

        # 返信のみを出力
        res = response.choices[0].message.content

        # 処理終了時刻
        end = time.perf_counter()

        # 経過時間
        elapsed_time_1 = end - start_1
        elapsed_time_2 = end - start_2
        elapsed_time_3 = end - start_3
        print(f"経過時間: {elapsed_time_1}秒")
        print(f"経過時間: {elapsed_time_2}秒")
        print(f"経過時間: {elapsed_time_3}秒")
        print(res)

        total_time_1 += elapsed_time_1
        total_time_2 += elapsed_time_2
        total_time_3 += elapsed_time_3

    print(f"平均経過時間: {total_time_1 / test_num}秒")
    print(f"平均経過時間: {total_time_2 / test_num}秒")
    print(f"平均経過時間: {total_time_3 / test_num}秒")

    return res


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
