# http://10.10.236.53:8000/v1/chat/completions   "model":"Qwen2.5-72B-Instruct-GPTQ-Int4"   这个部署了Qwen2.5 72B的int4量化

# 【实际后端对应的是两个模型服务】
# ID 2&3    http://10.10.185.2:11032/v1/chat/completions   "model":"Qwen2.5-72B-Instruct-GPTQ-Int4"
# ID 6&7    http://10.10.185.2:11034/v1/chat/completions   "model":"Qwen2.5-72B-Instruct-GPTQ-Int4"

# 先帮忙测试下功能上能不能满足百应业务需求

import json
from MA_RAG.test_get_token import get_knowledge_token
data = {
    "request_time": 11111111111111,
    "session_id": "600000009215",
    "assistant_id": 1,
    "assistant_name": "admin",
    "user_chat_input": "搜一下相关的竞品的情况，再结合刚刚问出来的专利的情况，优化一下刚才写的文章",
    "messages": [

        {
            "role": "user",
            "content": "我想了解yolo目标检测算法的发展历史，帮我将梳理各个版本的提升点，并按照版本号倒序排序整理到一个markdown表格中"
            # "content": "I want to know the development history of the Yolo object detection algorithm, help me sort out the improvement points of each version, and sort them into a Markdown table in reverse order of version number"
            # "content": "J'aimerais connaître l'historique du développement de l'algorithme de détection de cible Yolo, m'aider à trier les points de levage des différentes versions et à les trier dans un tableau markdown en ordre inverse par numéro de version"
        }
    ],
    # 浩鲸企业资料
    "dataset_list": [
        # {
        #     "datasetCode": "BYAI_DOC_1470205948423086080",
        #     "datasetDesc": "考勤助手知识库",
        #     "datasetId": 1478177085470244864,
        #     "datasetName": "考勤助手知识库",
        #     "resourceBizType": "KG_QA",
        #     "resourceId": 10010390
        # },
        {
            "datasetCode": "BYAI_KG_DOC_10151236",
            "datasetDesc": "qa",
            "datasetId": 1478186507211444224,
            "datasetName": "法律test",
            "resourceBizType": "KG_DOC",
            "resourceId": 10082222
        },
        {
            "datasetCode": "BYAI_KG_DOC_10185070",
            "datasetDesc": "qa",
            "datasetId": 1485090178244919296,
            "datasetName": "金融test",
            "resourceBizType": "KG_DOC",
            "resourceId": 10082222
        },
        {
            "datasetCode": "BYAI_KG_DOC_1474947679772663808",
            "datasetDesc": "qa",
            "datasetId": 1474947679772663808,
            "datasetName": "医疗test",
            "resourceBizType": "KG_DOC",
            "resourceId": 10082222
        },
        {
            "datasetCode": "BYAI_KG_DOC_10190927",
            "datasetDesc": "en_test",
            "datasetId": 1485813928160182272,
            "datasetName": "en_test",
            "resourceBizType": "KG_DOC",
            "resourceId": 10082222
        },
        {
            "datasetCode": "BYAI_KG_DOC_10352120",
            "datasetDesc": "教育test",
            "datasetId": 1497684059668955136,
            "datasetName": "qa",
            "resourceBizType": "KG_DOC",
            "resourceId": 10082222
        },
    ],
    "red_list": [
        "AGENT_418202816568",
        "DOC_418202817376",
        "AGENT_418203426476",
        "AGENT_418202898718",
        "AGENT_418203723146",
        "DOC_418202739337",
        "PLUGIN_418201752083",
        "DOC_418203366463",
        "DOC_418203679117",
        "AGENT_418203061255",
        "DOC_418203261945",
        "AGENT_418202743761",
        "AGENT_418203312107",
        "DOC_418203177877",
        "DOC_418203177910",
        "DOC_418202740596",
        "AGENT_418203657119",
        "DOC_418201806679",
        "DOC_418203177931",
        "DOC_418201806714",
        "AGENT_418203350543",
        "AGENT_418203402040",
        "AGENT_418203350247",
        "AGENT_418203356008",
        "AGENT_418201806660",
        "DOC_418203365523",
        "DOC_418203178002",
        "DOC_418201895245",
        "DOC_418202896228",
        "DOC_418203238697",
        "AGENT_418202816233",
        "AGENT_418203343776",
        "AGENT_418203417454",
        "DOC_418203342537",
        "DOC_418203177948",
        "DOC_418203177824",
        "DB_418201752083",
        "DOC_418203312208",
        "DOC_418203177966",
        "DOC_418203075051",
        "DOC_418203178019",
        "AGENT_418201752411",
        "DOC_418203177984",
        "DOC_418203177841",
        "DOC_418203177860",
        "AGENT_418202741717",
        "DOC_418203178036"
    ],
    "black_list": [
        "DOC_418202896228",
        "DB_418201752083",
        "DOC_418201806679",
        "DOC_418201806714",
        "AGENT_418201806660"
    ],
    "app_list": [
    ],
    # https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1
    # Qwen/Qwen3-235B-A22B
    # http://10.10.185.27:11111/v1
    # Qwen3-235B
    "env": {
        # "WHALEAGENT_API_SYSTEM": "",
        "WHALEAGENT_API_SYSTEM": "byai9e178831ca649cca55781aad9W",
        "LANGUAGE": "en_US",
        "RERANKER_BASE_URL": "http://10.10.185.13:7000/v1/rerank",
        "SEARCH_SERVICE_BASE_URL": "http://10.10.165.31:8084",
        "LLM_MODE_NAME_R": "Qwen/Qwen3-235B-A22B",
        "DIGIT_NUM_LIMIT": "30",
        "LLM_URL": "https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1",
        "LLM_TOKENS": "ailab_B5YNZRu+Zl3P0/RDPLCJkuGel3PyDlgYID2kYibg8xGjGdfXIETE/nSBSNynMNiyjJq8ERd33wSLpZSlMnY+/O3CCE8ZoUYrW5aj3wUu4fZo21AIYzNXJqM=",
        "LLM_MODE_NAME": "Qwen/Qwen3-235B-A22B",
        "WHALEAGENT_API_BASE": "http://10.10.165.30:2400/knowledge",
        "RERANKER_MODEL": "bge-reranker-v2-m3",
        "RERANKER_API_KEY": "wR518qmMWblBu0L1OSBdMnmdEgX1tO32ezndPJprQ8o",
        "SEARCH_NETWORK_BASE_URL": "http://10.10.195.6:17000/docchain",
        "TOKEN": "ailab_B5YNZRu+Zl3P0/RDPLCJkuGel3PyDlgYID2kYibg8xGjGdfXIETE/nSBSNynMNiyjJq8ERd33wSLpZSlMnY+/O3CCE8ZoUYrW5aj3wUu4fZo21AIYzNXJqM=",
        # "TOKEN": "ailab_R9G08Jakd2KMLtJYIdyculXyHxYSUMPqa1sNVWmm0dJ2sbwTM41G0EJeG73X4J4NCGng39D3A96RKxglC0XB4ni0r0iXLXnqjQS6TYD2UwY4qf5shD8ew64=",
        # "WHALEAGENT_API_KEY": get_knowledge_token().token,
        "WHALEAGENT_API_KEY": "SFFnK1dlZEFuUmFaQ2VSVnd0bXpYMlRCOGVsTzZSbk9HZDRNRWljNmlmYzVFMTJLL0M5Qm4zejB4L0Q2SlFJZFJ2bUUwVlp4Q1RuTVVuSUlMaXNpUzhvcS9zaGFtT0dORWhqY0UyK1FhZFY1Tjdzd2hLZzRrQkF0UGxrQVBXZHpXaE5KcU9YNm5IUXR3dGxTUnVUSk4rNHE0dTZCRGh0SWsxODVOMHFlZStjYmZ3bDdTN2dDU1J0dm9TTGNsU3VUNDgxNHRXYmpIYTlMVDk4UHpac2ladz09",
        "MINIO": {
            "accessKey": "minioadmin",
            "bucket": "byai",
            "endPoint": "http://10.10.196.92:2704",
            "secretKey": "minioadmin"
        },
        "MODEL_ANSWER_MESSAGE_ID": 600000014079,
        "LANGFUSE_PUBLIC_KEY": "pk-1234567890",
        "LANGFUSE_SECRET_KEY": "sk-1234567890",
        "LANGFUSE_HOST": "http://10.10.196.91:3000/langfuse",
        "assistant_id": 10000,
    },
    "chat_history": [

    ],
    "use_enterprise_information": True,
    "connect_net": False,
    "connect_params": {
        "header": {
            "X-Api-Key": "wZu9e2_z5HBvZsJxL7bnt6-RItyKMvo0eop0nHzrsU4"
        },
        "params": {
            "topic_id": "394"
        }
    },
    "agent_type": None,
    "third_agent_url": None,
    "third_agent_url_headers": None,
    "ext_params": None,
    "deep_think": False,
    "user_info": {
        "accessType": "SESSION",
        "assistantId": 1,
        "enterpriseId": 1,
        "paramMap": {},
        "sessionId": "6d22a85afd3-4ea4-4e76-a0fd-adaffsdff3",
        "userCode": "adminvip",
        "userId": 1,
        "userName": "平台管理员adminvip",
        "userStation": {
            "isAbroad": 1,
            "pStationId": 8,
            "stationId": 105,
            "stationIdPath": "-1.1.2.8.105",
            "stationName": "斯威士兰"
        },
        "usersOrganizations": [
            {
                "orgId": 1,
                "orgName": "百应客户",
                "pathCode": "-1.1",
                "positionId": 1,
                "positionName": "平台管理员",
                "userType": "PLAT_MAN"
            },
            {
                "orgId": 3167722223034483,
                "orgName": "浩鲸科技(一级经营委员会)",
                "pathCode": "-1.3167722223034483",
                "positionId": 3165966453330834,
                "positionName": "公司副总经理",
                "userType": "ORG_MAN"
            }
        ]
    }
}

