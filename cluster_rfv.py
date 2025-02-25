import streamlit as st # type: ignore
from streamlit_option_menu import option_menu # type: ignore
import pandas as pd
from PIL import Image
import timeit
import io
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Inicializa n_clusters no session_state
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3

# FUNÇÕES 
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def recencia_class(x, r, q_dict):
    if x <= q_dict[r][0.25]:
        return 'A'
    elif x <= q_dict[r][0.50]:
        return 'B'
    elif x <= q_dict[r][0.75]:
        return 'C'
    else:
        return 'D'

def freq_val_class(x, fv, q_dict):
    if x <= q_dict[fv][0.25]:
        return 'D'
    elif x <= q_dict[fv][0.50]:
        return 'C'
    elif x <= q_dict[fv][0.75]:
        return 'B'
    else:
        return 'A'

# FUNÇÕES RFV (adaptadas do código anterior)
def calcular_rfv(df):
    # Calcular Recência
    df_recencia = df.groupby('ID_cliente')['DiaCompra'].max().reset_index()
    df_recencia['Recencia'] = (pd.to_datetime('today') - df_recencia['DiaCompra']).dt.days

    # Calcular Frequência
    df_frequencia = df[['ID_cliente', 'CodigoCompra']].groupby('ID_cliente').count().reset_index()
    df_frequencia.rename(columns={'CodigoCompra': 'Frequencia'}, inplace=True)

    # Calcular Valor
    df_valor = df[['ID_cliente', 'ValorTotal']].groupby('ID_cliente').sum().reset_index()
    df_valor.rename(columns={'ValorTotal': 'Valor'}, inplace=True)

    # Merge dataframes
    df_rfv = pd.merge(df_recencia, df_frequencia, on='ID_cliente')
    df_rfv = pd.merge(df_rfv, df_valor, on='ID_cliente')
    return df_rfv

def segmentar_rfv(df_rfv):
    quartis = df_rfv[['Recencia', 'Frequencia', 'Valor']].quantile(q=[0.25, 0.5, 0.75])
    df_rfv['R_quartil'] = df_rfv['Recencia'].apply(recencia_class, args=('Recencia', quartis))
    df_rfv['F_quartil'] = df_rfv['Frequencia'].apply(freq_val_class, args=('Frequencia', quartis))
    df_rfv['V_quartil'] = df_rfv['Valor'].apply(freq_val_class, args=('Valor', quartis))
    df_rfv['RFV_Score'] = df_rfv[['R_quartil', 'F_quartil', 'V_quartil']].sum(axis=1) # Soma os valores dos quartis
    return df_rfv

def aplicar_kmeans(df_rfv, n_clusters):
    scaler = StandardScaler()
    rfv_scaled = scaler.fit_transform(df_rfv[['Recencia', 'Frequencia', 'Valor']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5) # n_init adicionado
    df_rfv['cluster'] = kmeans.fit_predict(rfv_scaled)
    return df_rfv

def filtrar_e_contar_clusters(df, n_cluster_max):
    """Filtra e exibe dados e contagem para clusters de 0 a n_cluster_max."""
    clusters_a_considerar = list(range(n_cluster_max + 1))  # Lista de clusters de 0 a n_cluster_max
    df_filtrado = df[df['cluster'].isin(clusters_a_considerar)].copy()
    contagem = df_filtrado.groupby('cluster').size().reset_index(name='count')
    st.write(f"## Dados dos Clusters 0 a {n_cluster_max}")
    st.dataframe(df_filtrado)
    st.write(f"## Contagem de Clientes nos Clusters 0 a {n_cluster_max}")
    st.dataframe(contagem)
    return df_filtrado, contagem

# CONFIGURAÇÕES DA PÁGINA 
st.set_page_config(
    page_title='RFV',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('# RFV')
st.markdown('---')

if 'menu' not in st.session_state:
    st.session_state.menu = 'Home'

if 'df_recencia' not in st.session_state:
    st.session_state.df_recencia = None
if 'df_frequencia' not in st.session_state:
    st.session_state.df_frequencia = None
if 'df_valor' not in st.session_state:
    st.session_state.df_valor = None

titulo_principal = """
        <div style="
            background-color: #000;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            width: 80%;
            margin: auto;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        ">
            <h1 style="color: #FFF;">RFV</h1>
            <p style="font-size: 16px; text-align: justify;">
                RFV significa recência, frequência e valor, e é utilizado para segmentação de clientes 
                baseado no comportamento de compras dos clientes. Esse método agrupa clientes em clusters 
                similares, permitindo ações de marketing e CRM mais direcionadas, personalizando conteúdos 
                e ajudando na retenção de clientes.
            </p>
            <h3>Para cada cliente, calculamos:</h3>
            <ul style="text-align: left;">
                <li><b>Recência (R):</b> Quantidade de dias desde a última compra.</li>
                <li><b>Frequência (F):</b> Quantidade total de compras no período.</li>
                <li><b>Valor (V):</b> Total de dinheiro gasto nas compras do período.</li>
            </ul>
        </div>
        """

# SIDEBAR
with st.sidebar:
    data_file_1 = st.file_uploader("INSIRA UM ARQUIVO PARA INICIARMOS A ANÁLISE", type=['csv', 'xlsx'])
    if data_file_1 is None:
        st.stop()
    df_compras = pd.read_csv(data_file_1)

    # Verifica se a coluna 'DiaCompra' existe
    if 'DiaCompra' in df_compras.columns:
        df_compras['DiaCompra'] = pd.to_datetime(df_compras['DiaCompra'], infer_datetime_format=True)
    else:
        st.error("A coluna 'DiaCompra' não está presente no arquivo.")
        st.stop()

    selected = option_menu(
        'Menu',
        ['Home','Recência (R)', 'Frequência (F)', 'Valor (V)', 'Análise RFV'],
        icons=['house', 'bar-chart-fill', 'bar-chart-fill', 'bar-chart-fill', 'bar-chart-fill'],
        menu_icon='cast',
        default_index=0,
        styles={
            'nav-link-selected': {'background-color': '#157806'},
        }
    )

if selected != st.session_state.menu:
    st.session_state.menu = selected
    st.rerun()
    
if selected == "Home":
    st.markdown(titulo_principal, unsafe_allow_html=True)

if selected == 'Recência (R)':
    st.markdown("<h1 style='font-size: 2em;'>Recência (R)</h1>", unsafe_allow_html=True) 
    st.markdown('Quantos dias faz que o cliente fez a sua última compra?')
    dia_atual = df_compras['DiaCompra'].max()
    df_recencia = df_compras.groupby('ID_cliente')['DiaCompra'].max().reset_index()
    df_recencia['Recencia'] = (dia_atual - df_recencia['DiaCompra']).dt.days
    st.session_state.df_recencia = df_recencia
    st.write(df_recencia.head(10))

if selected == 'Frequência (F)':
    st.markdown("<h1 style='font-size: 2em;'>Frequência (F)</h1>", unsafe_allow_html=True)
    st.markdown("Quantas vezes cada cliente comprou com a gente?")
    df_frequencia = df_compras.groupby('ID_cliente')['CodigoCompra'].count().reset_index()
    df_frequencia.columns = ['ID_cliente','Frequencia']
    st.session_state.df_frequencia = df_frequencia
    st.write(df_frequencia.head(10))

if selected == 'Valor (V)':
    st.markdown("<h1 style='font-size: 2em;'>Valor (V)</h1>", unsafe_allow_html=True)
    st.markdown("Qual o valor que cada cliente gastou no periodo ?")
    df_valor = df_compras.groupby('ID_cliente')['ValorTotal'].sum().reset_index()
    df_valor.columns = ['ID_cliente','Valor']
    st.session_state.df_valor = df_valor
    st.write(df_valor.head(10))

if selected == 'Análise RFV':
    st.markdown("<h1 style='font-size: 2em;'>Segmentação utilizando o RFV</h1>", unsafe_allow_html=True)
    st.write("""
             Uma forma eficaz de segmentar os clientes é através da criação de quartis para cada componente do RFV (Recência, Frequência e Valor). Nesse sistema, os clientes são classificados em quatro grupos: o melhor quartil recebe a letra 'A', o segundo melhor 'B', o terceiro 'C' e o pior 'D'.  
             A definição do melhor ou pior quartil varia conforme a métrica:  

            - **Recência (R):** Quanto menor a recência, melhor é o cliente, pois significa que ele comprou recentemente. Por isso, o menor quartil é classificado como 'A'.  
            - **Frequência (F):** Neste caso, a lógica se inverte: quanto maior a frequência de compras, melhor é o cliente. Assim, o maior quartil recebe a letra 'A'.  

            Essa classificação facilita a criação de estratégias direcionadas, ajudando a identificar os clientes mais valiosos e aqueles que precisam de ações específicas para aumentar seu engajamento.
             """
    )

    df_RF = pd.merge(st.session_state.df_recencia, st.session_state.df_frequencia, on='ID_cliente')
    df_RFV = pd.merge(df_RF, st.session_state.df_valor, on='ID_cliente')
    st.write('## ')

    st.write('Quartis para o RFV')
    quartis = df_RFV.quantile(q=[0.25,0.5,0.75])
    st.write(quartis)

    st.write('Tabela após a criação dos grupos')
    df_RFV['R_quartil'] = df_RFV['Recencia'].apply(recencia_class, args=('Recencia', quartis))
    df_RFV['F_quartil'] = df_RFV['Frequencia'].apply(freq_val_class, args=('Frequencia', quartis))
    df_RFV['V_quartil'] = df_RFV['Valor'].apply(freq_val_class, args=('Valor', quartis))
    df_RFV['RFV_Score'] = (df_RFV.R_quartil + df_RFV.F_quartil + df_RFV.V_quartil)
    st.write(df_RFV.head())

    st.write('Quantidade de clientes por grupos')
    st.write(df_RFV['RFV_Score'].value_counts())

    st.write('#### Clientes com menor recência, maior frequência e maior valor gasto')
    st.write(df_RFV[df_RFV['RFV_Score']=='AAA'].sort_values('Valor', ascending=False).head(10))

    st.write('### Ações de marketing/CRM')

    dict_acoes = {'AAA': 'Enviar cupons de desconto, Pedir para indicar nosso produto pra algum amigo, Ao lançar um novo produto enviar amostras grátis pra esses.',
        'DDD': 'Churn! clientes que gastaram bem pouco e fizeram poucas compras, fazer nada',
        'DAA': 'Churn! clientes que gastaram bastante e fizeram muitas compras, enviar cupons de desconto para tentar recuperar',
        'CAA': 'Churn! clientes que gastaram bastante e fizeram muitas compras, enviar cupons de desconto para tentar recuperar'
        }

    df_RFV['acoes de marketing/crm'] = df_RFV['RFV_Score'].map(dict_acoes)
    st.write(df_RFV.head())

    st.write('Quantidade de clientes por tipo de ação')
    st.write(df_RFV['acoes de marketing/crm'].value_counts(dropna=False))

    st.markdown("### RFV com Cluster")
    n_clusters_slider = st.slider("Número de clusters para K-Means", 1, 5, st.session_state.n_clusters)
    st.session_state.n_clusters = n_clusters_slider # Salva o valor atualizado

    df_rfv = calcular_rfv(df_compras)
    df_rfv_segmentado = segmentar_rfv(df_rfv)

    # Aplica o KMeans com o número de clusters DEFINIDO no slider
    df_rfv_cluster = aplicar_kmeans(df_rfv_segmentado.copy(), n_clusters_slider)
    st.session_state.df_rfv_cluster = df_rfv_cluster

    # Filtrar e exibir dados dos clusters de 0 ao valor selecionado no slider
    df_cluster_filtrado, contagem_cluster = filtrar_e_contar_clusters(df_rfv_cluster, n_clusters_slider)

    df_xlsx = to_excel(df_RFV)
    st.download_button(label='📥 Download', data=df_xlsx, file_name='RFV_.xlsx')