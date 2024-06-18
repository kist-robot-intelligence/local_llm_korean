# local_llm_korean

## 코드 소개

main_streaming.py는 스트리밍 형태로 결과를 출력하는 모델입니다.

api_server_multi.py와 api_client_multi.py는 함께 실행되는 파일입니다.
api_server_multi.py는 여러 개의 client의 context를 각각 처리하는 역할을 합니다.
다만 현재는 동시에 두 개 이상의 client가 server에 질의를 보내게 되면 두 개의 응답이 섞이는 문제가 있습니다.
