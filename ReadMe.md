![header](https://capsule-render.vercel.app/api?type=rect&color=gradient&text=데이터%20제작%20프로젝트%20&fontSize=45)
<div align="left">
	<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" />
	<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white" />
</div>
&nbsp;

# Members
- **김도윤**  : 데이터 EDA, 데이터 검수, 데이터 Augmentation
- **김윤호**  : 데이터 검수, 데이터 Augmentation
- **김종해**  : 데이터 검수, 외부 데이터 EDA, 학습결과 시각화
- **조재효**  : 데이터 검수, 데이터 Augmentation, Valid Set 구성
- **허진녕**  : 데이터 검수, 데이터 Augmentation

&nbsp;

# 프로젝트 개요
> 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다.
또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다.  
이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다. <sup>[[1]](#footnote_1)</sup>

&nbsp;

# 프로젝트 수행 절차
<h3> 1. ICDAR17_Kor 데이터셋의 EDA 진행  </h3>
<h3> 2. wandb 연결 및 random seed 고정  </h3>
<h3> 3. EAST 모델을 이용한 관통테스트 및 기준성능 확인  </h3>
<h3> 4. 외부 데이터셋 공수 및 데이터 Augmentation 진행  </h3>

&nbsp;

# 문제정의
<h3> 1. 주어진 데이터의 부족  </h3>  

- 기존에 주어진 ICDAR17_Kor 데이터 수는 약 300장으로, 학습을 진행하기엔 너무 적은 양이라고 판단하였다.  
- 이를 보완하기 위해, 직접 Annotation을 진행한 실습 자료와 ICDAR17의 모든 데이터를 활용하여 학습을 진행하였다.

 <h3> 2-1. 데이터의 Annotation 오류  </h3>  

- Annotation 실습 데이터와 ICDAR17_Kor를 학습으로 사용하기 전, 이미지와 글자 영역을 시각화하여 직접 확인한 결과, 일부 이미지에서 글자 영역이 누락되거나 잘못된 영역으로 표현되는 등 데이터의 일관성을 해치는 요소가 존재하였다.
- 이를 보완하고자 팀원 모두가 일정 데이터만큼 분담하여 같은 가이드라인 하에 데이터 검수 및 정제를 진행하였다.

<h3> 2-2. Polygon의 전처리  </h3>  

- 기존에 주어진 ICDAR17_Kor와는 달리, Annotation 실습 데이터의 글자영역은 4개의 점보다 더 많은 점을 사용하여 Polygon 형태로 표현한 경우가 있었는데, 이는 주어진 baseline 코드와는 맞지 않아 학습을 진행할 수 없었다.
- 이를 해결하고자, Polygon으로 표현된 글자영역은 해당 영역을 포함하는 가장 작은 직사각형으로 변환하여 표현하였다.
- 하지만 직사각형으로의 변환으로 인해 기존에 글자영역이 아니었던 부분이 글자영역으로 편입되어 데이터의 노이즈를 발생시켰고, 학습에 방해가 되었다.
- 결국 Polygon 형태의 데이터가 소수라는 점을 근거로, 4개의 점으로 글자영역을 표현한 데이터만을 학습에 사용하도록 결정하였다.

<h3> 3. Validation Set 생성  </h3>  

- 기존에 주어진 baseline 코드는 mean loss, angle loss 등 학습 시의 loss만을 산출할 수 있었고, 이러한 지표는 모델의 성능을 표현하기엔 역부족이었다.
- 모델의 성능을 측정하면서 overfitting 여부까지 확인하기 위해, 기존에 학습으로 사용하던 ICDAR17_Kor와 Annotation 실습 데이터셋의 20%를 Validation Set으로 선정하고, 학습 과정에서 Validation Set에 대한 f1-score를 확인할 수 있도록 구조를 설계하였다.

&nbsp;


# Reference
<a name="footnote_1">[1]</a> 출처 : 위키피디아