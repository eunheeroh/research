# 최지영_논문분석   2025년 7월2일일
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import re
import io
import base64
import matplotlib.font_manager as fm
import warnings
import os
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="연구자 관계도 분석 시스템",
    page_icon="📊",
    layout="wide"
)

# 한글 폰트 설정 개선
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # Windows 한글 폰트 경로 직접 지정
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',  # 맑은 고딕
            'C:/Windows/Fonts/gulim.ttc',   # 굴림
            'C:/Windows/Fonts/dotum.ttc',   # 돋움
            'C:/Windows/Fonts/batang.ttc',  # 바탕
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"한글 폰트 설정 완료: {font_path}")
                    return font_prop
            except:
                continue
        
        # 폰트 파일이 없으면 시스템 폰트에서 찾기
        font_list = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Dotum', 'Batang']
        for font_name in font_list:
            try:
                font_prop = fm.FontProperties(family=font_name)
                if font_prop.get_name() != 'DejaVu Sans':
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"시스템 폰트 설정 완료: {font_name}")
                    return font_prop
            except:
                continue
        
        # 마지막 수단: 기본 설정
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        print("기본 폰트 설정 사용")
        return None
        
    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return None

# 폰트 설정 실행
font_prop = setup_korean_font()

class ResearchAnalyzer:
    def __init__(self):
        self.data = None
        self.researchers = []
        self.keywords = []
        self.graph = nx.Graph()
        
    def load_data(self, data):
        """데이터 로드"""
        if isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            self.data = data
        return self.data
    
    def preprocess_keywords(self, keyword_column):
        """주제어 전처리"""
        if keyword_column not in self.data.columns:
            raise ValueError(f"컬럼 '{keyword_column}'을 찾을 수 없습니다.")
            
        self.data['processed_keywords'] = self.data[keyword_column].apply(
            lambda x: [kw.strip().lower() for kw in re.split(r'[,;]', str(x)) if kw.strip()]
        )
        
        all_keywords = []
        for keywords in self.data['processed_keywords']:
            all_keywords.extend(keywords)
        
        self.keywords = all_keywords
        return len(set(all_keywords))
        
    def analyze_researcher_relationships(self, researcher_column):
        """연구자 간 관계 분석"""
        if researcher_column not in self.data.columns:
            raise ValueError(f"컬럼 '{researcher_column}'을 찾을 수 없습니다.")
            
        researcher_keywords = {}
        for idx, row in self.data.iterrows():
            researchers = str(row[researcher_column]).split(',')
            keywords = row['processed_keywords']
            
            for researcher in researchers:
                researcher = researcher.strip()
                if researcher not in researcher_keywords:
                    researcher_keywords[researcher] = []
                researcher_keywords[researcher].extend(keywords)
        
        self.researchers = list(researcher_keywords.keys())
        
        for i, researcher1 in enumerate(self.researchers):
            for j, researcher2 in enumerate(self.researchers[i+1:], i+1):
                keywords1 = set(researcher_keywords[researcher1])
                keywords2 = set(researcher_keywords[researcher2])
                
                intersection = len(keywords1.intersection(keywords2))
                union = len(keywords1.union(keywords2))
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > 0:
                        self.graph.add_edge(researcher1, researcher2, weight=similarity)
        
        return len(self.graph.edges())
    
    def analyze_keyword_network(self):
        """키워드 네트워크 분석"""
        keyword_graph = nx.Graph()
        
        for keywords in self.data['processed_keywords']:
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i+1:]:
                    if keyword_graph.has_edge(kw1, kw2):
                        keyword_graph[kw1][kw2]['weight'] += 1
                    else:
                        keyword_graph.add_edge(kw1, kw2, weight=1)
        
        return keyword_graph

def plot_to_base64(fig):
    """matplotlib 그래프를 base64로 인코딩"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str

def create_korean_chart(fig, ax, title, xlabel, ylabel):
    """한글 차트 생성 헬퍼 함수"""
    # 폰트 재설정
    setup_korean_font()
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 축 레이블 폰트 설정
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(10)
    
    plt.tight_layout()

def main():
    st.title("📊 연구자 관계도 분석 시스템")
    st.markdown("---")
    
    # 사이드바 - 데이터 입력
    st.sidebar.header("📋 데이터 입력")
    
    # 샘플 데이터 제공
    if st.sidebar.checkbox("샘플 데이터 사용"):
        sample_data = [
            {
                'title': '딥러닝을 활용한 이미지 분류 연구',
                'researchers': '김철수, 이영희, 박민수',
                'keywords': '딥러닝, 이미지분류, CNN, 컴퓨터비전',
                'year': 2023
            },
            {
                'title': '자연어처리 기반 감정분석 시스템',
                'researchers': '이영희, 최지원',
                'keywords': '자연어처리, 감정분석, BERT, 텍스트마이닝',
                'year': 2022
            },
            {
                'title': '강화학습을 이용한 게임 AI 개발',
                'researchers': '박민수, 김철수',
                'keywords': '강화학습, 게임AI, 딥러닝, Q-learning',
                'year': 2021
            },
            {
                'title': '빅데이터 분석을 통한 고객 행동 예측',
                'researchers': '최지원, 정수진',
                'keywords': '빅데이터, 고객행동, 예측모델, 머신러닝',
                'year': 2020
            },
            {
                'title': '컴퓨터비전 기반 자율주행 시스템',
                'researchers': '정수진, 김철수',
                'keywords': '컴퓨터비전, 자율주행, 딥러닝, 이미지처리',
                'year': 2023
            }
        ]
        
        data = pd.DataFrame(sample_data)
        st.sidebar.success("샘플 데이터가 로드되었습니다!")
        
    else:
        # 파일 업로드
        uploaded_file = st.sidebar.file_uploader(
            "CSV 파일 업로드", 
            type=['csv'],
            help="논문 제목, 연구자, 키워드, 발행년도(year) 컬럼이 포함된 CSV 파일을 업로드하세요."
        )
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("파일이 성공적으로 업로드되었습니다!")
        else:
            st.info("샘플 데이터를 사용하거나 CSV 파일을 업로드해주세요.")
            return
    
    # 컬럼 선택
    st.sidebar.header("🔧 설정")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        title_col = st.selectbox("논문 제목 컬럼", data.columns.tolist(), index=0)
    with col2:
        researcher_col = st.selectbox("연구자 컬럼", data.columns.tolist(), index=1)
    
    keyword_col = st.sidebar.selectbox("키워드 컬럼", data.columns.tolist(), index=2)
    year_col = st.sidebar.selectbox("발행년도 컬럼", data.columns.tolist(), index=3 if 'year' in data.columns else 0)
    
    # 분석 실행 버튼
    if st.sidebar.button("🚀 분석 시작", type="primary"):
        with st.spinner("데이터를 분석하고 있습니다..."):
            # 분석기 초기화
            analyzer = ResearchAnalyzer()
            
            # 데이터 로드 및 전처리
            analyzer.load_data(data)
            unique_keywords = analyzer.preprocess_keywords(keyword_col)
            connections = analyzer.analyze_researcher_relationships(researcher_col)
            
            # 결과 표시
            st.success(f"분석 완료! {len(data)}개 논문, {len(analyzer.researchers)}명 연구자, {unique_keywords}개 키워드, {connections}개 연결")
            
            # 메인 컨텐츠 영역
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 기본 통계")
                
                # 통계 카드
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("논문 수", len(data))
                with metric_col2:
                    st.metric("연구자 수", len(analyzer.researchers))
                with metric_col3:
                    st.metric("키워드 수", unique_keywords)
                
                # 데이터 미리보기 (발행년도 포함)
                st.subheader("📋 데이터 미리보기")
                st.dataframe(data[[title_col, researcher_col, keyword_col, year_col]].head(), use_container_width=True)
            
            with col2:
                st.subheader("🏆 주요 키워드")
                
                # 키워드 빈도 분석
                keyword_counts = Counter(analyzer.keywords)
                top_keywords = keyword_counts.most_common(10)
                
                # 키워드 차트
                fig, ax = plt.subplots(figsize=(8, 6))
                keywords, counts = zip(*top_keywords)
                bars = ax.barh(range(len(keywords)), counts, color='skyblue', alpha=0.7)
                ax.set_yticks(range(len(keywords)))
                ax.set_yticklabels(keywords)
                
                create_korean_chart(fig, ax, '상위 10개 키워드 빈도', '빈도', '키워드')
                ax.invert_yaxis()
                
                # 값 표시
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           str(count), ha='left', va='center')
                
                st.pyplot(fig)
            
            # 연구자 관계도
            st.subheader("👥 연구자 관계도")
            
            if len(analyzer.graph.nodes()) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 노드 크기 (연결 수에 비례)
                node_sizes = [analyzer.graph.degree(node) * 200 for node in analyzer.graph.nodes()]
                
                # 엣지 두께 (유사도에 비례)
                edge_weights = [analyzer.graph[u][v]['weight'] * 5 for u, v in analyzer.graph.edges()]
                
                # 레이아웃 설정
                pos = nx.spring_layout(analyzer.graph, k=2, iterations=50)
                
                # 그래프 그리기
                nx.draw_networkx_nodes(analyzer.graph, pos, 
                                     node_size=node_sizes, 
                                     node_color='lightblue',
                                     alpha=0.8)
                
                nx.draw_networkx_edges(analyzer.graph, pos, 
                                     width=edge_weights,
                                     alpha=0.6,
                                     edge_color='gray')
                
                # 한글 폰트 설정 후 라벨 그리기
                if font_prop:
                    nx.draw_networkx_labels(analyzer.graph, pos, 
                                          font_size=12,
                                          font_weight='bold',
                                          font_family=font_prop.get_name())
                else:
                    nx.draw_networkx_labels(analyzer.graph, pos, 
                                          font_size=12,
                                          font_weight='bold')
                
                create_korean_chart(fig, ax, "연구자 간 관계도 (키워드 유사도 기반)", "", "")
                plt.axis('off')
                st.pyplot(fig)
                
                # 연구자별 상세 정보
                st.subheader("🔍 연구자별 상세 정보")
                
                researcher_info = []
                for researcher in analyzer.researchers:
                    degree = analyzer.graph.degree(researcher)
                    connections = [neighbor for neighbor in analyzer.graph.neighbors(researcher)]
                    researcher_info.append({
                        '연구자': researcher,
                        '연결 수': degree,
                        '연결된 연구자': ', '.join(connections) if connections else '없음'
                    })
                
                researcher_df = pd.DataFrame(researcher_info)
                researcher_df = researcher_df.sort_values('연결 수', ascending=False)
                st.dataframe(researcher_df, use_container_width=True)
                
            else:
                st.warning("연구자 간 연결이 없습니다.")
            
            # 키워드 네트워크
            st.subheader("🏷️ 키워드 네트워크")
            
            min_weight = st.slider("최소 연결 강도", min_value=1, max_value=5, value=1)
            
            keyword_graph = analyzer.analyze_keyword_network()
            filtered_edges = [(u, v, d) for u, v, d in keyword_graph.edges(data=True) 
                             if d['weight'] >= min_weight]
            
            if filtered_edges:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                filtered_graph = nx.Graph()
                # 가중치를 숫자로 변환하여 추가
                for u, v, d in filtered_edges:
                    filtered_graph.add_edge(u, v, weight=float(d['weight']))
                
                # 노드 크기 (연결 수에 비례)
                node_sizes = [filtered_graph.degree(node) * 100 for node in filtered_graph.nodes()]
                
                # 엣지 두께 (가중치에 비례)
                edge_weights = [filtered_graph[u][v]['weight'] for u, v in filtered_graph.edges()]
                
                # 레이아웃 설정
                pos = nx.spring_layout(filtered_graph, k=3, iterations=100)
                
                # 그래프 그리기
                nx.draw_networkx_nodes(filtered_graph, pos, 
                                     node_size=node_sizes, 
                                     node_color='lightcoral',
                                     alpha=0.8)
                
                nx.draw_networkx_edges(filtered_graph, pos, 
                                     width=edge_weights,
                                     alpha=0.6,
                                     edge_color='red')
                
                # 키워드 라벨에도 한글 폰트 적용
                if font_prop:
                    nx.draw_networkx_labels(filtered_graph, pos, 
                                          font_size=10,
                                          font_weight='bold',
                                          font_family=font_prop.get_name())
                else:
                    nx.draw_networkx_labels(filtered_graph, pos, 
                                          font_size=10,
                                          font_weight='bold')
                
                create_korean_chart(fig, ax, f"키워드 네트워크 (최소 연결 강도: {min_weight})", "", "")
                plt.axis('off')
                st.pyplot(fig)
                
            else:
                st.warning(f"가중치 {min_weight} 이상의 키워드 연결이 없습니다.")
            
            # 상세 분석 리포트
            st.subheader("📊 상세 분석 리포트")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔗 연결 강도별 분포**")
                if analyzer.graph.edges():
                    weights = [analyzer.graph[u][v]['weight'] for u, v in analyzer.graph.edges()]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(weights, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
                    create_korean_chart(fig, ax, '연구자 간 연결 강도 분포', '연결 강도 (Jaccard 유사도)', '연결 수')
                    st.pyplot(fig)
            
            with col2:
                st.markdown("**📈 키워드 사용 패턴**")
                keyword_freq = pd.DataFrame(keyword_counts.most_common(20), 
                                          columns=['키워드', '빈도'])
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(keyword_freq['키워드'], keyword_freq['빈도'], 
                       marker='o', color='orange', linewidth=2, markersize=6)
                create_korean_chart(fig, ax, '키워드 빈도 분포', '키워드', '빈도')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

if __name__ == "__main__":
    main() 