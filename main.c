#include "AI.h"
#include <stdio.h>
extern double c;
int main(){
	double sum_sq_error = 0.0; //�������� ��
	double avg_sq_error=0.0;//������ ���
	double test_accurancy;
	int tr;
	printf("####################################################################\n");
	printf("    [������Ʈ : \"Neural Network\" �� �̿��� ���� �ν� �ý��� ���� ]	\n");
	printf("							 2013253083\n");
	printf("							     ���ֿ�\n");
	printf("####################################################################\n");
	printf("####################################################################\n\n");
	printf("������ ���� �� : ");
	for (int i = 0; i < NLayer; i++){
		printf("%d ", M[i]);
	}
	printf("\n1epoch=%d\n", N_tr_examples);
	printf("train_threshold : %f\n", train_threshold);
	printf("�ʱ� c��: %f\n", c);
	printf("---------------------------------------\n");
	printf("Ʈ���̴� ������ �� �׽�Ʈ ������ �о���� ��...\n");
	data_Onmemory();
	printf("���� ����ġ ���� ��...\n ");
	weight_init();
	printf("Ʈ���̴� ����\n");
	tr=training();
	test_accurancy=test();
	printf("------------------------------------------\n");
	printf("��� ���:\n");
	for (int i = 0; i < NLayer; i++){
		printf("%d ", M[i]);
	}
	printf("\n���� c : %f\n", c);
	printf("1epoch = %d\n", N_tr_examples);
	printf("�ݺ��� ���� �� : %d\n" , tr);
	printf("test_accurancy : %f\%", test_accurancy*100);

	return 0;
}