#ifndef _AI_H_
#define _AI_H_

#define NLayer 3
#define MLayerSize 1000
#define m0 100
#define m1 200
#define m2 10
#define N 784+1 
#define train_threshold 0.005
#define N_tr_examples 60000
#define N_te_examples 10000

int M[NLayer];


void data_Onmemory(); //트레이닝 데이터 및 테스트 데이터 메모리에 저장하는 함수
void weight_init(); // 초기 가중치 설정 함수
void forward_compute(); 
void backward_compute();
void weight_update();
int training(); //트레이닝 데이터로 훈련하는 함수
double test(); //테스트 데이터로 정답률 출력 함수
int correctNum(); // 출력 값중에 가장 큰 값을 가지는 배열의 index 출력
#endif 