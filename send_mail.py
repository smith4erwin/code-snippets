# -*- coding: utf-8 -*-



import smtplib
from email.mime.text import MIMEText
from email.header import Header


def get_message(content, subobject, fromm, to):
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = fromm
    message['To'] = to
    message['subobject'] = Header(subobject, 'utf-8')
    return message


if __name__ == '__main__':
    pass
    sender = 'locke.lee@qq.com'


#   smtp = smtplib.STMP()
#   stmp.connect(mail_host, 25)
#   smtp.loggin(mail_user, mail_passwd)

#   sender = 'locke.lee@qq.com'
#   maai
#   smtp.send()
    
