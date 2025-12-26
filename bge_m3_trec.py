import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BGE-M3-Baseline")

@dataclass
class ExperimentConfig:
    name: str
    weights: Dict[str, float]
    top_k: Dict[str, int]
    rrf_k: int

@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    score: float

class BGERetriever:
    def __init__(self, dataset_name: str, device: str = None):
        self.dataset_name = dataset_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.index = None
        self.corpus = None
        self.ids = None
        self.doc_ids = None

    def load_model(self):
        logger.info(f"Đang tải BAAI/bge-m3 trên thiết bị: {self.device}...")
        self.model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        logger.info("Load BGE-M3 thành công!")

    def load_data(self):
        logger.info(f"Đang tải dataset: {self.dataset_name}")
        ds = load_dataset(self.dataset_name)
        data = ds["train"]
        
        self.corpus = data["chunk_text"]
        self.ids = data["chunk_id"]
        self.doc_ids = data["doc_id"]
        logger.info(f"Load {len(self.corpus)} văn bản.")

    def build_index(self, batch_size=4):
        logger.info("Đang mã hóa (Encoding) Corpus...")
        embeddings = self.model.encode(
            self.corpus,
            batch_size=batch_size, 
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True 
        )

        logger.info("Building FAISS Index...")
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Inner Product = Cosine
        logger.info(f"Index xong {self.index.ntotal} vector.")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Encode Query
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search FAISS
        scores, indices = self.index.search(q_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append(SearchResult(
                chunk_id=self.ids[idx],
                doc_id=self.doc_ids[idx],
                score=float(score)
            ))
        return results

class ExperimentRunner:
    TOPIC_DATA = [
        {
            "topic-id": 1,
            "query": "coronavirus origin",
            "question": "what is the origin of COVID-19",
            "narrative": "seeking range of information about the SARS-CoV-2 virus's origin, including its evolution, animal source, and first transmission into humans"
        },
        {
            "topic-id": 2,
            "query": "coronavirus response to weather changes",
            "question": "how does the coronavirus respond to changes in the weather",
            "narrative": "seeking range of information about the SARS-CoV-2 virus viability in different weather/climate conditions as well as information related to transmission of the virus in different climate conditions"
        },
        {
            "topic-id": 3,
            "query": "coronavirus immunity",
            "question": "will SARS-CoV2 infected people develop immunity? Is cross protection possible?",
            "narrative": "seeking studies of immunity developed due to infection with SARS-CoV2 or cross protection gained due to infection with other coronavirus types"
        },
        {
            "topic-id": 4,
            "query": "how do people die from the coronavirus",
            "question": "what causes death from Covid-19?",
            "narrative": "Studies looking at mechanisms of death from Covid-19."
        },
        {
            "topic-id": 5,
            "query": "animal models of COVID-19",
            "question": "what drugs have been active against SARS-CoV or SARS-CoV-2 in animal studies?",
            "narrative": "Papers that describe the results of testing drugs that bind to spike proteins of the virus or any other drugs in any animal models. Papers about SARS-CoV-2 infection in cell culture assays are also relevant."
        },
        {
            "topic-id": 6,
            "query": "coronavirus test rapid testing",
            "question": "what types of rapid testing for Covid-19 have been developed?",
            "narrative": "Looking for studies identifying ways to diagnose Covid-19 more rapidly."
        },
        {
            "topic-id": 7,
            "query": "serological tests for coronavirus",
            "question": "are there serological tests that detect antibodies to coronavirus?",
            "narrative": "Looking for assays that measure immune response to COVID-19 that will help determine past infection and subsequent possible immunity."
        },
        {
            "topic-id": 8,
            "query": "coronavirus under reporting",
            "question": "how has lack of testing availability led to underreporting of true incidence of Covid-19?",
            "narrative": "Looking for studies answering questions of impact of lack of complete testing for Covid-19 on incidence and prevalence of Covid-19."
        },
        {
            "topic-id": 9,
            "query": "coronavirus in Canada",
            "question": "how has COVID-19 affected Canada",
            "narrative": "seeking data related to infections (confirm, suspected, and projected) and health outcomes (symptoms, hospitalization, intensive care, mortality)"
        },
        {
            "topic-id": 10,
            "query": "coronavirus social distancing impact",
            "question": "has social distancing had an impact on slowing the spread of COVID-19?",
            "narrative": "seeking specific information on studies that have measured COVID-19's transmission in one or more social distancing (or non-social distancing) approaches"
        },
        {
            "topic-id": 11,
            "query": "coronavirus hospital rationing",
            "question": "what are the guidelines for triaging patients infected with coronavirus?",
            "narrative": "Seeking information on any guidelines for prioritizing COVID-19 patients infected with coronavirus based on demographics, clinical signs, serology and other tests."
        },
        {
            "topic-id": 12,
            "query": "coronavirus quarantine",
            "question": "what are best practices in hospitals and at home in maintaining quarantine?",
            "narrative": "Seeking information on best practices for activities and duration of quarantine for those exposed and/ infected to COVID-19 virus."
        },
        {
            "topic-id": 13,
            "query": "how does coronavirus spread",
            "question": "what are the transmission routes of coronavirus?",
            "narrative": "Looking for information on all possible ways to contract COVID-19 from people, animals and objects"
        },
        {
            "topic-id": 14,
            "query": "coronavirus super spreaders",
            "question": "what evidence is there related to COVID-19 super spreaders",
            "narrative": "seeking range of information related to the number and proportion of super spreaders, their patterns of behavior that lead to spread, and potential prevention strategies targeted specifically toward super spreaders"
        },
        {
            "topic-id": 15,
            "query": "coronavirus outside body",
            "question": "how long can the coronavirus live outside the body",
            "narrative": "seeking range of information on the SARS-CoV-2's virus's survival in different environments (surfaces, liquids, etc.) outside the human body while still being viable for transmission to another human"
        },
        {
            "topic-id": 16,
            "query": "how long does coronavirus survive on surfaces",
            "question": "how long does coronavirus remain stable on surfaces?",
            "narrative": "Studies of time SARS-CoV-2 remains stable after being deposited from an infected person on everyday surfaces in a household or hospital setting, such as through coughing or touching objects."
        },
        {
            "topic-id": 17,
            "query": "coronavirus clinical trials",
            "question": "are there any clinical trials available for the coronavirus",
            "narrative": "seeking specific COVID-19 clinical trials ranging from trials in recruitment to completed trials with results"
        },
        {
            "topic-id": 18,
            "query": "masks prevent coronavirus",
            "question": "what are the best masks for preventing infection by Covid-19?",
            "narrative": "What types of masks should or should not be used to prevent infection by Covid-19?"
        },
        {
            "topic-id": 19,
            "query": "what alcohol sanitizer kills coronavirus",
            "question": "what type of hand sanitizer is needed to destroy Covid-19?",
            "narrative": "Studies assessing chemicals and their concentrations needed to destroy the Covid-19 virus."
        },
        {
            "topic-id": 20,
            "query": "coronavirus and ACE inhibitors",
            "question": "are patients taking Angiotensin-converting enzyme inhibitors (ACE) at increased risk for COVID-19?",
            "narrative": "Looking for information on interactions between coronavirus and angiotensin converting enzyme 2 (ACE2) receptors, risk for patients taking these medications, and recommendations for these patients."
        },
        {
            "topic-id": 21,
            "query": "coronavirus mortality",
            "question": "what are the mortality rates overall and in specific populations",
            "narrative": "Seeking information on COVID-19 fatality rates in different countries and in different population groups based on gender, blood types, or other factors"
        },
        {
            "topic-id": 22,
            "query": "coronavirus heart impacts",
            "question": "are cardiac complications likely in patients with COVID-19?",
            "narrative": "Seeking information on the types, frequency and mechanisms of cardiac complications caused by coronavirus."
        },
        {
            "topic-id": 23,
            "query": "coronavirus hypertension",
            "question": "what kinds of complications related to COVID-19 are associated with hypertension?",
            "narrative": "seeking specific outcomes that hypertensive (any type) patients are more/less likely to face if infected with the virus"
        },
        {
            "topic-id": 24,
            "query": "coronavirus diabetes",
            "question": "what kinds of complications related to COVID-19 are associated with diabetes",
            "narrative": "seeking specific outcomes that diabetic (any type) patients are more/less likely to face if infected with the virus"
        },
        {
            "topic-id": 25,
            "query": "coronavirus biomarkers",
            "question": "which biomarkers predict the severe clinical course of 2019-nCOV infection?",
            "narrative": "Looking for information on biomarkers that predict disease outcomes in people infected with coronavirus, specifically those that predict severe and fatal outcomes."
        },
        {
            "topic-id": 26,
            "query": "coronavirus early symptoms",
            "question": "what are the initial symptoms of Covid-19?",
            "narrative": "Studies of patients and the first clinical manifestations they develop upon active infection?"
        },
        {
            "topic-id": 27,
            "query": "coronavirus asymptomatic",
            "question": "what is known about those infected with Covid-19 but are asymptomatic?",
            "narrative": "Studies of people who are known to be infected with Covid-19 but show no symptoms?"
        },
        {
            "topic-id": 28,
            "query": "coronavirus hydroxychloroquine",
            "question": "what evidence is there for the value of hydroxychloroquine in treating Covid-19?",
            "narrative": "Basic science or clinical studies assessing the benefit and harms of treating Covid-19 with hydroxychloroquine."
        },
        {
            "topic-id": 29,
            "query": "coronavirus drug repurposing",
            "question": "which SARS-CoV-2 proteins-human proteins interactions indicate potential for drug targets. Are there approved drugs that can be repurposed based on this information?",
            "narrative": "Seeking information about protein-protein interactions for any of the SARS-CoV-2 structural proteins that represent a promising therapeutic target, and the drug molecules that may inhibit the virus and the host cell receptors at entry step."
        },
        {
            "topic-id": 30,
            "query": "coronavirus remdesivir",
            "question": "is remdesivir an effective treatment for COVID-19",
            "narrative": "seeking specific information on clinical outcomes in COVID-19 patients treated with remdesivir"
        },
        {
            "topic-id": 31,
            "query": "difference between coronavirus and flu",
            "question": "How does the coronavirus differ from seasonal flu?",
            "narrative": "Includes studies ranging from those focusing on genomic differences to global public health impacts, but must draw direct comparisons between COVID-19 and seasonal influenza."
        },
        {
            "topic-id": 32,
            "query": "coronavirus subtypes",
            "question": "Does SARS-CoV-2 have any subtypes, and if so what are they?",
            "narrative": "Papers that discuss subtypes of the virus, from named subtypes to speculative subtypes based on genomic or geographic clustering."
        },
        {
            "topic-id": 33,
            "query": "coronavirus vaccine candidates",
            "question": "What vaccine candidates are being tested for Covid-19?",
            "narrative": "Seeking studies that discuss possible, but specific, COVID-19 vaccines. Includes articles from those describing the mechanisms of action of specific proposed vaccines to actual clinical trials, but excluding articles that do not name a specific vaccine candidate."
        },
        {
            "topic-id": 34,
            "query": "coronavirus recovery",
            "question": "What are the longer-term complications of those who recover from COVID-19?",
            "narrative": "Seeking information on the health outcomes for those that recover from the virus. Excludes studies only focusing on adverse effects related to a particular COVID-19 drug."
        },
        {
            "topic-id": 35,
            "query": "coronavirus public datasets",
            "question": "What new public datasets are available related to COVID-19?",
            "narrative": "Seeking articles that specifically release new data related to SARS-CoV-2 or COVID-19, including genomic data, patient data, public health data, etc. Articles that reference previously existing datasets are not relevant."
        },
        {
            "topic-id": 36,
            "query": "SARS-CoV-2 spike structure",
            "question": "What is the protein structure of the SARS-CoV-2 spike?",
            "narrative": "Looking for studies of the structure of the spike protein on the virus using any methods, such as cryo-EM or crystallography"
        },
        {
            "topic-id": 37,
            "query": "SARS-CoV-2 phylogenetic analysis",
            "question": "What is the result of phylogenetic analysis of SARS-CoV-2 genome sequence?",
            "narrative": "Looking for a range of studies which provide the results of phylogenetic network analysis on the SARS-CoV-2 genome"
        },
        {
            "topic-id": 38,
            "query": "COVID inflammatory response",
            "question": "What is the mechanism of inflammatory response and pathogenesis of COVID-19 cases?",
            "narrative": "Looking for a range of studies which describes the inflammatory response cells and pathogenesis during the Coronavirus Disease 2019 (COVID-19) outbreak, including the mechanism of anti-inflammatory drugs, corticosteroids, and vitamin supplements"
        },
        {
            "topic-id": 39,
            "query": "COVID-19 cytokine storm",
            "question": "What is the mechanism of cytokine storm syndrome on the COVID-19?",
            "narrative": "Looking for studies that describes mechanism of development of cytokine storm syndrome among COVID-19 cases and the range of drugs used for the therapy of cytokine storm"
        },
        {
            "topic-id": 40,
            "query": "coronavirus mutations",
            "question": "What are the observed mutations in the SARS-CoV-2 genome and how often do the mutations occur?",
            "narrative": "Looking for studies that describes the emergence of genomic diversity of the coronavirus due to recurrent mutations which explore the potential genomic site of the mutation, mechanisms and its potential or observed clinical implications in the pathogenicity of the virus"
        }
    ]

    EXPERIMENT_CONFIGS = [
        ExperimentConfig(
            name="baseline_rrf60",
            weights={"query": 1.0, "question": 2.5, "narrative": 0.7},
            top_k={"query": 200, "question": 300, "narrative": 500},
            rrf_k=60
        ),
        ExperimentConfig(
            name="rrf20",
            weights={"query": 1.0, "question": 2.5, "narrative": 0.7},
            top_k={"query": 200, "question": 300, "narrative": 500},
            rrf_k=20
        ),
        ExperimentConfig(
            name="rrf100",
            weights={"query": 1.0, "question": 2.5, "narrative": 0.7},
            top_k={"query": 200, "question": 300, "narrative": 500},
            rrf_k=100
        ),
        ExperimentConfig(
            name="question_strict",
            weights={"query": 0.8, "question": 3.5, "narrative": 0.2},
            top_k={"query": 150, "question": 400, "narrative": 100},
            rrf_k=30
        ),
        ExperimentConfig(
            name="question_soft",
            weights={"query": 1.0, "question": 3.0, "narrative": 0.3},
            top_k={"query": 150, "question": 400, "narrative": 150},
            rrf_k=80
        ),
        ExperimentConfig(
            name="narrative_recall",
            weights={"query": 1.0, "question": 2.5, "narrative": 0.15},
            top_k={"query": 200, "question": 300, "narrative": 800},
            rrf_k=80
        ),
        ExperimentConfig(
            name="anti_noise",
            weights={"query": 1.0, "question": 2.5, "narrative": 0.1},
            top_k={"query": 200, "question": 300, "narrative": 100},
            rrf_k=120
        ),
        ExperimentConfig(
            name="minimal",
            weights={"query": 1.0, "question": 3.0, "narrative": 0.0},
            top_k={"query": 200, "question": 300, "narrative": 0},
            rrf_k=60
        ),
    ]
    
    def __init__(self, retriever: BGERetriever):
        self.retriever = retriever

    def run_experiment(self, config: ExperimentConfig, output_dir: str):
        logger.info(f"Chạy thí nghiệm: {config.name}")
        rows = []
        
        for item in self.TOPIC_DATA:
            topic_id = item["topic-id"]
            fields = {
                "query": item["query"],
                "question": item["question"],
                "narrative": item["narrative"]
            }
            doc_score = defaultdict(float)
            
            for field, text in fields.items():
                if config.top_k[field] == 0 or config.weights[field] == 0: continue
                results = self.retriever.search(text, top_k=config.top_k[field])
                weight = config.weights[field]
                for rank, result in enumerate(results, start=1):
                    doc_score[result.doc_id] += weight / (config.rrf_k + rank)
            
            ranked_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(ranked_docs[:10], start=1):
                rows.append({"topic-id": topic_id, "cord-id": doc_id})
        
        df = pd.DataFrame(rows)
        output_file = f"{output_dir}/submission_BGE_{config.name}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Đã lưu kết quả tại: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Chạy Baseline BGE-M3")
    parser.add_argument("--output-dir", type=str, default="./results_bge")
    parser.add_argument("--dataset", type=str, default="manhngvu/cord19_chunked_300_words")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    retriever = BGERetriever(args.dataset)
    retriever.load_model()
    retriever.load_data()
    retriever.build_index(batch_size=args.batch_size)

    runner = ExperimentRunner(retriever)
    
    target_config = next((c for c in runner.EXPERIMENT_CONFIGS if c.name == "baseline_rrf60"), None)
    if target_config:
        runner.run_experiment(target_config, args.output_dir)

if __name__ == "__main__":
    main()