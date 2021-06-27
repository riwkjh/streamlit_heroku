import streamlit as st
from PIL import Image
from tasks.classification import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("바다 어류 분류 딥러닝모델 데모")
st.markdown("### 1. 간단한 설명")
st.markdown(
    "kaggle의 [a large scale fish dataset]( https://www.kaggle.com/crowww/a-large-scale-fish-dataset)로 만들어본 9종류의 바다어류를 판정하는 딥러닝 모델입니다. 모델의 성능은 테스트 데이터셋에 대해서 정확도 99.3%입니다. 단, out-of-distribution과 같은 아웃라이어를 전혀 고려하지 않은 수치이므로 학습 데이터셋의 환경과 다른 환경에서 촬영된 물고기 사진을 보여줄 경우에는 모델이 정확하게 인식하지 못하는 경우도 생깁니다.")
st.markdown("### 2. 대응 어류종류")
st.markdown("* Black Sea sprat")
st.markdown("* Gilt-Head Bream (귀족도미)")
st.markdown("* Hourse Mackerel (전갱이)")
st.markdown("* Red Mullet (붉은 숭어)")
st.markdown("* Red Sea Bream (참돔)")
st.markdown("* Sea Bass (농어)")
st.markdown("* Shrimp (새우)")
st.markdown("* Striped Red Mullet")
st.markdown("* Trout (송어)")
st.markdown("### 3. 사용법")
st.markdown("바다물고기의 사진을 업로드 해주세요. 딥러닝 모델이 각각의 어류에 해당할 확률을 계산해서 결과를 출력합니다.")

file_up = st.file_uploader("물고기 jpg사진을 올려주세요", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("딥러닝 모델이 사진을 처리하고 있습니다....")
    labels = predict(file_up)

    st.write("예측결과는.....")
    # print out the top 5 prediction labels with scores
    for i, label in enumerate(labels):
        st.write("{}위: ".format(i), " ",
                 label[0], " ", "확률: ", "{0:0.2f} %".format(label[1]))


st.text("만든이 : 김종혁")