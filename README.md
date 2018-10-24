# Neural-Network-
[  프로젝트:  "Neural Network 기반 숫자인식 시스템 개발 ]

o 개발 시스템 설명

우리가 디자인할 neural network 은 다음과 같은 구조를 가지도록 한다:


- 입력으로 들어 오는 feature 는 N 개 ( N 은 상수기호로 정의됨) 이다. 결국 총 입력 시그날의 수는 N+1 이다.
       
- 전체 층(layer)의 수는 3 개이다.각 층의 뉴론의 수는 기호상수  m0, m1, m2 로 선언함.

- 츨력층의 뉴론의 수 = m2

우리의 시스템은 문자인식 시스템이다. 가능한 문자 집합은 { 0, 1, 2, …, 9} 로 한다.

[1]   Training data : 
    - 인공지능 연구집단에서 많이 사용하는 MNIST data 를 이용한다.
       traindata.txt,  testdata.txt 의 두 개의 파일로 구성된다.
       전자는 훈련에 후자는 테스트에 이용된다.
    - 각 예제는 28 X 28 의 숫자 이미지와 이의 정답레이블 (0 ~9 중 한 수) 로 구성된다.
    - 이미지의 각 셀의 값은 0 ~ 255 사이의 정수이다. 이것은 해당 셀의 gray scale 이다.
    - 우리는 각 셀 값을  0 ~ 1 사이의 실수로 변환하여 사용한다.
    - 데이터는 파일 train.txt 와 test.txt 에 주어진다. 전자는 6만개의 예제, 후자는 1만개의 예제로 
       를 가진다.
-	각 예제는 다음과 같은 format 를 가진다.
-	첫 라인:   4  ( target label 을 나타낸다. 즉 이미지의 정답 레이블을 나타낸다.)
-	다음 28 개의 라인: 각 라인은 28 개의 정수를 가진다.  28X 28 의  이미지 정보를 나타낸다.
-	

[3] 시스템의 구성은 다음과 같다.
전체 시스템은 training part 와 testing part 로 구성되어 있다.
(가) training part
    -  initialization :
       먼저 모든 weight parameter 들을 초기화 해야 한다. 
    - 한 epoch 의 구성
      (a) 6만개 각 훈련 예를 하나씩 취해 가면서 다음 작업을 수행한다:
          .   forword computation
                모든 뉴론의 s, f 를 순방향으로 계산한다.
         .   backward computation
                모든 뉴론의 delta를 역방향으로 계산한다.
         .   weight update
                모든 뉴론의  weights  를 delta 를 이용하여 갱신한다.
         위 3 단계를 모든 훈련예제에 대하여 수행하고 나면 한 epoch 를 수행한 것이 된다.
      (b) 한 epoch 를 수행한 후에 평균에러 (average squared error ) 를 다음 과정을 거쳐 구한다.
         .  각 훈련 예마다 forward computation 을 수행하여 출력층 뉴론의 출력마다 f 를 계산한다.
             각 출력층 뉴론 출력의 squared error 를 계산한다.
         .  모든 출력층 뉴론 출력의 squared error 를 합한다 ( squared error of an example )
         .  위의  모든 훈련예의 squared error of a example 을 모두 합한다.
         .  모든 훈련예에 대하여 위 작업을 수행한 결과로 total sum of squared error 을 구한다.
         .  결국 평균 에러(average squared error) 는 다음과 같이 구한다:
                    total sum of squared error / (훈련예의 수 * 출력층 뉴론의 수) 
      (c)  다음 epoch 로 더 나아갈 지를 결정한다.  예를 들어 train_threshold = 0.001 을 이용한다.
              만약 average squared error  <  train_threshold 이면, 
              종료할지 여부를 물어보아 계속하라면 종료하지 않고 다음 epoch 를 수행한다.
              종료하라고 하면 다음 epoch 로 가지 않고, test 과정으로 간다.

 (나)  testing part 
    - 1만개의 테스트 예제 각각에 대하여 다음을 수행한다.
    - 테스팅 과정에서는 각 뉴론의 함수를  sigmoid 대신 다시 threshold 함수로 교체하여야 하지만 
우리 과제에서는 그냥 sigmoid 함수를 이용한다. 출력 층의 여러 뉴론 중  f 가 가장 큰 것에
 해당하는 문자를 답으로 인식한다. 

    - 각각의 테스트 예제에 대하여 출력을 계산한다. 이 출력이 이 테스트 예제의 target label 과
       동일하면 이 테스트 예제는 답을 맞춘 것이고 아니면 못 맞춘 것이다.
       이 작업을 모든 훈련 예제에 대하여 시행한다.
       그 다음  accuracy 를 다음과 같이 계산한다:

               test accuracy =  정답을 맞춘 예의수/전체테스트수

   
o  실험 방법:
(가)	 위에서 설명한 시스템을 개발한다. 그리고 test accuracy 를 출력하여 본다 .
        만약 test accuracy 가 좋지 않다고 생각되면  다음 사항 을 변경하여 다시 실험 (즉 훈련-테스트) 과정을
시도해 본다:
       -  각 층의 뉴론의 수
       -  학습률 c 
  (니)  실험결과로 출력할 정보:  
       각층의 뉴론 갯수, 학습률 c,  훈련에 소요된 epoch 수, test accuracy.


