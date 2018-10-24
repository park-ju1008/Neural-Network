#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include "AI.h"

int M[NLayer] = { m0, m1, m2 };
int trainData[N_tr_examples][N - 1] = { 0, };
int d_tr[N_tr_examples][m2] = { 0, };
int testData[N_te_examples][N - 1] = { 0, };
int d_te[N_te_examples][m2] = { 0, };

int input[N];
int D[m2];


double s[NLayer][MLayerSize];
double f[NLayer][MLayerSize];
double delta[NLayer][MLayerSize];
double W[NLayer][MLayerSize][MLayerSize];
double c = 0.05;

void data_Onmemory(){
	FILE *trainfp;
	FILE *testfp;
	int i=0, j, ans;

	trainfp = fopen("train.txt", "rt");
	testfp = fopen("test.txt", "rt");
	if (trainfp != NULL){
		
		while (!feof(trainfp)){
				fscanf(trainfp, "%d ", &ans);
				d_tr[i][ans] = 1;
				for (j = 0; j < N-1; j++){
					fscanf(trainfp, "%d ", &trainData[i][j]);
				}
				i++;
		}
		fclose(trainfp);
		
	}
	if (testfp != NULL){
		while (!feof(testfp)){
			for (i = 0; i < N_te_examples; i++){
				fscanf(testfp, "%d", &ans);
				d_te[i][ans] = 1;
				for (j = 0; j < N - 1; j++){
					fscanf(testfp, "%d ", &testData[i][j]);
				}

			}
		}
		fclose(testfp);
	}
}


void weight_init(){
	int i,j,k,r,pre_layer;
	srand(time(NULL));
	for (i = 0; i < NLayer; i++){
		if (i == 0){
			pre_layer = N;
			for (j = 0; j < M[i]; j++){
				for (k = 0; k < pre_layer; k++){
					r = (double)(rand());
					W[i][j][k] = (r / (double)RAND_MAX)-0.5; //�Է� ���� Ŀ w�� �۰� ����
				}
			}
		}
			else{
				pre_layer = M[i - 1] + 1;
				for (j = 0; j < M[i]; j++){
					for (k = 0; k < pre_layer; k++){
						r = (double)(rand());
						W[i][j][k] = (r / (double)RAND_MAX)-0.5;
					}
				}
			}
	}
}

void forward_compute(){
	int i, j, layer;
	//0���� ���� s��� �� f���
	for (i = 0; i < M[0]; i++){
		s[0][i] = 0.0;
		for (j = 0; j < N; j++){
			s[0][i] += (input[j]/255.0) * W[0][i][j];
		}
		f[0][i] = 1.0 / (1.0 + exp(-s[0][i]));
	}
	f[0][m0] = 1.0;
	//��1 ���� s��� �� f���
	for (layer = 1; layer < NLayer; layer++){
		for (i = 0; i < M[layer]; i++){
			s[layer][i] = 0.0; //�ʱ�ȭ
			for (j = 0; j < M[layer - 1] + 1; j++){
				s[layer][i] += f[layer-1][j] * W[layer][i][j];//layer���� i��° ��
			}
			f[layer][i] = 1.0 / (1.0 + exp(-s[layer][i])); //layer���� i��° ������ ��°�
		}
		f[layer][M[layer]] = 1.0; //���� �Է�
	}
}
void backward_compute(){
	int i,j;
	double tsum;
	int k = NLayer - 1; //��������
	for (i = 0; i < M[k]; i++){ //���������� ��Ÿ���� ���ϱ�
		delta[k][i] = (D[i] - f[k][i])*f[k][i] * (1 - f[k][i]);
	}
	for (k = NLayer - 2; k >= 0; k--){//�߰����� ��Ÿ�� ���ϱ�
		for (i = 0; i < M[k]; i++){ //k�� ���� ��Ÿ ��
			tsum = 0.0;
			for (j = 0; j < M[k + 1]; j++){//k+1�� ������ ��Ÿ*����ġ
				tsum += delta[k + 1][j] * W[k + 1][j][i];
			}
			delta[k][i] = f[k][i] * (1 - f[k][i])*tsum;
		}
	}
}
void weight_update(){
	int i, j, layer;

	for (i = 0; i < M[0]; i++){// 0���� ����ġ ������Ʈ
		for (j = 0; j < N; j++){
			W[0][i][j] += c*delta[0][i] * input[j];
		}
	}
	for (layer = 1; layer < NLayer; layer++){// layer ���� ����ġ ������Ʈ
		for (i = 0; i < M[layer]; i++){
			for (j = 0; j < M[layer - 1] + 1; j++){
				W[layer][i][j] += c*delta[layer][i] * f[layer - 1][j];
			}
		}
	}
	
}

int training(){
	int tr = 0, i, j;
	int epoch = N_tr_examples;
	double sum_sq_error;
	double avg_sq_error;
	char flag = 'n';
	while (flag == 'n' || flag == 'N'){ //���� ������ �̸��� ��� ���� 
		do{
			for (i = 0; i < epoch; i++){
				memcpy(input, trainData[i], sizeof(trainData[i]));
				input[N - 1] = 1;
				memcpy(D, d_tr[i], sizeof(d_tr[i]));
				forward_compute();
				backward_compute();
				weight_update();
			}
			sum_sq_error = 0.0;

			for (i = 0; i < epoch; i++){

				memcpy(input, trainData[i], sizeof(trainData[i]));
				input[N - 1] = 1;
				memcpy(D, d_tr[i], sizeof(d_tr[i]));
				forward_compute();
				for (j = 0; j < M[NLayer - 1]; j++)
					sum_sq_error += (D[j] - f[NLayer - 1][j])* (D[j] - f[NLayer - 1][j]);

			}
			c = c*0.98;
			tr++;
			avg_sq_error = sum_sq_error / (epoch*M[NLayer - 1]); //������ ���
			printf("#%depoch �� ��� ������: %f \n", tr, avg_sq_error);
		} while (avg_sq_error > train_threshold); //������ ������ ���� ���� ���� �ݺ�
		printf("��ǥ �������� �����߽��ϴ�. �Ʒ��� ��ġ�ðڽ��ϱ�? (y/n):");
		fflush(stdin);
		scanf("%c", &flag);
	}
	return tr;
}

double test(){
	int num_correct = 0;
	int i,index;
	double test_accurancy;
	for (i = 0; i < N_te_examples; i++){
		memcpy(input, testData[i], sizeof(testData[i]));
		input[N - 1] = 1;
		memcpy(D, d_te[i], sizeof(d_te[i]));
		forward_compute();
		index = correctNum();
		if (D[index] == 1){
			num_correct++;
		}
		
	}
	test_accurancy = (double)num_correct / N_te_examples;
	return test_accurancy;
}

int correctNum(){
	double temp=f[NLayer-1][0];
	int ret = 0;
	for (int i = 1; i < M[NLayer - 1]; i++){
		if (temp < f[NLayer - 1][i]){
			temp = f[NLayer - 1][i];
			ret = i;
		}
	}
	return ret; //���� ū ���� �迭�� index ���
}