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


void data_Onmemory(); //Ʈ���̴� ������ �� �׽�Ʈ ������ �޸𸮿� �����ϴ� �Լ�
void weight_init(); // �ʱ� ����ġ ���� �Լ�
void forward_compute(); 
void backward_compute();
void weight_update();
int training(); //Ʈ���̴� �����ͷ� �Ʒ��ϴ� �Լ�
double test(); //�׽�Ʈ �����ͷ� ����� ��� �Լ�
int correctNum(); // ��� ���߿� ���� ū ���� ������ �迭�� index ���
#endif 