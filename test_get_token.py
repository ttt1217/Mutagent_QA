
import json
import hashlib
import random
import time
import requests
import logging
from typing import Optional, Dict, Any


class AuthParams:
    def __init__(self):
        self.secret = None
        self.system_key = None
        self.system_secret = None
        self.nonce = None
        self.timestamp = None
        self.signature = None
        self.user_code = None
        self.user_source = None
        self.acc_com_id = None

    def to_dict(self):
        return {
            'secret': self.secret,
            'systemKey': self.system_key,
            'systemSecret': self.system_secret,
            'nonce': self.nonce,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'userCode': self.user_code,
            'userSource': self.user_source,
            'accComId': self.acc_com_id
        }


class KnowledgeToken:
    def __init__(self):
        self.system_key = None
        self.token = None

    def to_dict(self):
        return {
            'systemKey': self.system_key,
            'token': self.token
        }


class CurrentSessionUser:
    @staticmethod
    def get_current_user_id():
        # 需要根据实际情况实现
        return "your_user_id"

    @staticmethod
    def get_current_user_code():
        # 需要根据实际情况实现
        return "your_user_code"

    @staticmethod
    def get_enterprise_id():
        # 需要根据实际情况实现
        return "your_enterprise_id"


class Config:
    @staticmethod
    def get_system_properties(key):
        # 需要根据实际情况实现
        return "your_knowledge_base_url"


def md5_encrypt(signature_str: str, secret: str) -> str:
    """
    MD5加密函数，需要根据原Java代码中MD5Utils.encrypt的具体实现来调整
    """
    # 这里假设是简单的MD5加密，可能需要根据实际的加密逻辑调整
    combined = signature_str + "{" + secret + "}"
    # combined = f"{signature_str} +'}'"{secret}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def get_knowledge_token() -> Optional[KnowledgeToken]:
    try:
        # system_open_man
        secret = "null"
        system_key = "byai9e178831ca649cca55781aad9W"
        system_secret = "byaihceMUw="
        nonce = random.randint(-9223372036854775808, 9223372036854775807)  # Java long范围
        timestamp = str(int(time.time() * 1000))  # 毫秒时间戳
        user_code = "adminvip"
        user_source = "byai"
        acc_com_id = "1"

        # 获取URL
        url = 'http://10.10.196.92:19900/knowledge/knowledgeService/support/user/v1/getToken'
        # {"accComId": "1", "nonce": 752801263529226640, "secret": "null",
        #  "signature": "3099c125027c544c03116ad31d9280ed", "systemKey": "byai9e178831ca649cca55781aad9W",
        #  "systemSecret": "byaihceMUw=", "timestamp": "1748437914427", "userCode": "adminvip", "userSource": "byai"}
        # 生成签名
        signature_str = system_key + ":" + system_secret + ":" + timestamp + ":" + str(nonce)

        # signature_str = f"byai9e178831ca649cca55781aad9W:byaihceMUw=:1748437914427:752801263529226640"
        signature = md5_encrypt(signature_str, secret)

        # 创建认证参数
        auth_params = AuthParams()
        auth_params.secret = secret
        auth_params.system_key = system_key
        auth_params.system_secret = system_secret
        auth_params.nonce = nonce
        auth_params.timestamp = timestamp
        auth_params.signature = signature
        auth_params.user_code = user_code
        auth_params.user_source = user_source
        auth_params.acc_com_id = acc_com_id

        logging.info(f"当前认证入参: {json.dumps(auth_params.to_dict(), ensure_ascii=False)}")

        # 发送HTTP请求
        # headers = {
        #     'Content-Type': 'application/json; charset=utf-8'
        # }

        response = requests.post(
            url,
            json=auth_params.to_dict(),
        )
        response.raise_for_status()  # 检查HTTP错误

        # 解析响应
        json_response = response.json()
        result_object = json_response.get("resultObject", {})

        # 创建KnowledgeToken对象
        knowledge_token = KnowledgeToken()
        knowledge_token.system_key = system_key
        knowledge_token.token = result_object.get("token")

        logging.info(f"current token: {json.dumps(knowledge_token.to_dict(), ensure_ascii=False)}")
        return knowledge_token

    except Exception as e:
        logging.error(f"获取知识库token失败: {str(e)}", exc_info=True)
        return None


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 获取token
    token = get_knowledge_token()
    if token:
        print(f"获取到token: {token.token}")
    else:
        print("获取token失败")
