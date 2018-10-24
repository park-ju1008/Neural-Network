#include "AI.h"
#include <stdio.h>
extern double c;
int main(){
	double sum_sq_error = 0.0; //오차들의 합
	double avg_sq_error=0.0;//오차의 평균
	double test_accurancy;
	int tr;
	printf("####################################################################\n");
	printf("    [프로젝트 : \"Neural Network\" 을 이용한 숫자 인식 시스템 개발 ]	\n");
	printf("							 2013253083\n");
	printf("							     박주영\n");
	printf("####################################################################\n");
	printf("####################################################################\n\n");
	printf("각층의 뉴런 수 : ");
	for (int i = 0; i < NLayer; i++){
		printf("%d ", M[i]);
	}
	printf("\n1epoch=%d\n", N_tr_examples);
	printf("train_threshold : %f\n", train_threshold);
	printf("초기 c값: %f\n", c);
	printf("---------------------------------------\n");
	printf("트레이닝 데이터 및 테스트 데이터 읽어오는 중...\n");
	data_Onmemory();
	printf("랜덤 가중치 설정 중...\n ");
	weight_init();
	printf("트레이닝 시작\n");
	tr=training();
	test_accurancy=test();
	printf("------------------------------------------\n");
	printf("출력 결과:\n");
	for (int i = 0; i < NLayer; i++){
		printf("%d ", M[i]);
	}
	printf("\n최종 c : %f\n", c);
	printf("1epoch = %d\n", N_tr_examples);
	printf("반복한 에폭 수 : %d\n" , tr);
	printf("test_accurancy : %f\%", test_accurancy*100);

	return 0;
}