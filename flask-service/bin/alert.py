# -*- coding: utf-8 -*-

import sys
import time
import hashlib
import requests


app_name = "inforec"
secret = "bb9b752c-4cd4-4d69-9340-f93ad0922808"


def get_sign(timestamp):
    return hashlib.md5((secret + str(timestamp)).encode('utf-8')).hexdigest()


def get_params(content, to):
    timestamp = int(time.time())
    signature = get_sign(timestamp)
    return {
       "ac_appName"  : app_name,
       "ac_timestamp": timestamp,
       "ac_signature": signature,
       "content"     : content,
       "to"          : to,
       "isSync"      : "1",
    }


def send_email(content, email, title='cbir'):
    params = get_params(content, email)
    params["title"] = title
    resp = requests.post("http://alarm.netease.com/api/sendEmail", data=params)
    return resp.json()


def send_POPO(content, popo):
    params = get_params(content, popo)
    resp = requests.post("http://alarm.netease.com/api/sendPOPO", data=params)
    return resp.json()


def send_SMS(content, mobile):
    params = get_params(content, mobile)
    resp = requests.post("http://alarm.netease.com/api/sendSMS", data=params)
    return resp.json()


def send_phone(content, mobile):
    params = get_params(content, mobile)
    resp = requests.post("http://alarm.netease.com/api/sendVoice", data=params)
    return resp.json()


def send_message(message):
    try:
        send_SMS(message, "15623341190")
    except Exception as e:
        print(e)
        try:
            send_POPO(message, "lilinhan@corp.netease.com")
        except Exception as e:
            print(e)
            print("send message fail")

    #send_email(message, "lilinhan@corp.netease.com")
    #send_phone(message, "15623341190") 


if __name__ == "__main__":
    message = sys.argv[1]
    send_message(message)


