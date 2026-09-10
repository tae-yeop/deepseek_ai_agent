원리가 llm 모델 연결
그래프를 만들면 app이 나옴
    각 그래프 노드는 state를 받고 결과를 리턴하는 함수로 이뤄짐
초기 state를 만듬
app.invoke(init_state) 이렇게 넣으면 그래프를 타면서 동작
타는 순서는 edge를 어떻게 등록했냐에 따라
