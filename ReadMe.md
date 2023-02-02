![header](https://capsule-render.vercel.app/api?type=rect&color=0:FFD700,100:DDA0DD&text=데이터%20제작%20프로젝트%20&fontSize=45)
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
> 카메라로 카드를 인식하거나 주차장에서 차량 번호를 인식하는 일은 더 이상 익숙하지 않은 일이 아닙니다. 이미지 속의 문자를 컴퓨터가 인식하도록 하는 OCR 기술은 우리의 일상에 다가왔고, 그 수준 또한 상당히 높습니다. 이러한 기술력이 만들어지기까지, 데이터가 어떤 중요도를 갖는지 체감해볼 것입니다.

&nbsp;

# 프로젝트 수행 절차
<h3> 1. 주어진 데이터의 EDA 진행  </h3>
<h3> 2. 주어진 데이터에 대한 EAST 모델의 기준성능 측정  </h3>
<h3> 3. 데이터 Annotation 검수 및 보완  </h3>
<h3> 4. 외부 데이터셋 공수 및 데이터 Augmentation 진행  </h3>

&nbsp;

# 대회 제약
- EAST 모델의 글자 검출 성능을 높이는 것만으로 대회를 진행
- 외부 데이터 수집 및 데이터 Annotation 수정은 허용되나, 모델 수정은 불허

&nbsp;

# 문제정의
<h3> 1. 주어진 데이터의 부족  </h3>  

- 기존에 주어진 ICDAR17_Kor 데이터 수는 약 300장으로, 학습을 진행하기엔 너무 적은 양이라고 판단하였다.  
- 이를 보완하기 위해, 직접 Annotation을 진행한 실습 자료와 ICDAR17의 모든 데이터를 활용하여 학습을 진행하였다.

 <h3> 2. 데이터의 Annotation 오류  </h3>  

- Annotation 실습 데이터와 ICDAR17_Kor를 학습으로 사용하기 전, 이미지와 글자 영역을 시각화하여 직접 확인한 결과, 일부 이미지에서 글자 영역이 누락되거나 잘못된 영역으로 표현되는 등 데이터의 일관성을 해치는 요소가 존재하였다.
- 이를 보완하고자 팀원 모두가 일정 데이터만큼 분담하여 같은 가이드라인 하에 데이터 검수 및 정제를 진행하였다.

<h3> 3. Polygon의 전처리  </h3>  

- 기존에 주어진 ICDAR17_Kor와는 달리, Annotation 실습 데이터의 글자영역은 4개의 점보다 더 많은 점을 사용하여 Polygon 형태로 표현한 경우가 있었는데, 이는 주어진 모델에 맞지 않아 학습을 진행할 수 없었다.
- 이를 해결하고자, Polygon으로 표현된 글자영역은 해당 영역을 포함하는 가장 작은 직사각형으로 변환하여 표현하였다.
- 하지만 직사각형으로의 변환으로 인해 기존에 글자영역이 아니었던 부분이 글자영역으로 편입되어 데이터의 노이즈를 발생시켰고, 학습에 방해가 되었다.
- 결국 Polygon 형태의 데이터가 소수라는 점을 근거로, 4개의 점으로 글자영역을 표현한 데이터만을 학습에 사용하도록 결정하였다.

&nbsp;