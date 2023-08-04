# MLstudy_taehun

졸업논문 작성을 위한 머신러닝 공부 및 실습

## pendulum
비선형 미분방정식인 진자운동방정식 해석, 중력만 가해지는 경우(pendulum.ipynb), 공기저항 등의 항력이 존재하는 경우(pendulum2.ipynb)로 나눔

## neural ODE
(solver1D 1st, 2nd DiffEq)

주피터 노트북 상에서 monitor1D 통한 학습과정이 실시간으로 갱신되어 보이지 않음. (원인 불명)
하지만 .py 파일을 컴파일해서 실행하면 확인가능.

더해서 이 코드로 훈련되지 않는 2nd ode가 있음. 이유는 모르겠지만 추측컨대 y=e^x 꼴 함수에서 이런 일이 생기는 것으로 보임.
