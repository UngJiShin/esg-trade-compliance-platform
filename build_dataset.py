import os
import sys
import shutil
import json

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, 'data')
os.makedirs(data_dir, exist_ok=True)

# Direct relative paths avoiding BlockC
explicit_sources = {
    'master_dataset.csv': '4일차 학생교안/4일차 학생교안/master_dataset.csv',
    'new_products_sample.csv': '4일차 학생교안/4일차 학생교안/new_products_sample.csv',
    'shipping_docs_sample.csv': '4일차 학생교안/4일차 학생교안/선적서류_샘플.csv',
    'screening_result.csv': '4일차 학생교안/4일차 학생교안/screening_result.csv',
    'today_review_report.csv': '4일차 학생교안/4일차 학생교안/오늘의_검토리포트.csv',
    'dashboard_integrated.csv': '5일차_실습자료_학생배포용/5일차_실습자료/02_대시보드_데이터/대시보드_연동데이터.csv',
    'dashboard_test_sample.csv': '5일차_실습자료_학생배포용/5일차_실습자료/02_대시보드_데이터/신규샘플_통합테스트.csv',
    'master_country_esg.csv': '5일차_실습자료_학생배포용/5일차_실습자료/02_대시보드_데이터/마스터_국가별_ESG리스크.csv',
    'master_tariff_cbam.csv': '5일차_실습자료_학생배포용/5일차_실습자료/02_대시보드_데이터/마스터_품목별_관세율.csv',
    'manager_mapping.csv': '5일차_실습자료_학생배포용/5일차_실습자료/02_대시보드_데이터/담당자_매핑표.csv',
    'weekly_risk_log.csv': '5일차_실습자료_학생배포용/5일차_실습자료/03_심화예제/주간리스크_감지로그.csv',
    'rag_questions.csv': '5일차_실습자료_학생배포용/5일차_실습자료/03_심화예제/RAG_청크실험_질의.csv',
    'multilingual_rules.csv': '5일차_실습자료_학생배포용/5일차_실습자료/03_심화예제/다국어_규정_번역표.csv',
    'cleaning_rules.csv': '2일차 실습예제/2일차 실습예제/cleaning_rules.csv',
    'risk_category_ref.csv': '2일차 실습예제/2일차 실습예제/risk_category_ref.csv',
    'sample_invoice.xlsx': '2일차 실습예제/2일차 실습예제/인보이스_샘플.xlsx',
    'sample_export_declaration.xlsx': '2일차 실습예제/2일차 실습예제/수출신고필증_샘플.xlsx',
    'reason_docs.csv': '2일차 실습예제/2일차 실습예제/reason_docs.csv',
    'risk_keywords.csv': '학생용 실습예제파일/1일차 학생용 실습예제파일/risk_keywords.csv',
    'risk_threshold.csv': '학생용 실습예제파일/1일차 학생용 실습예제파일/risk_threshold.csv',
    'sample_docs.csv': '학생용 실습예제파일/1일차 학생용 실습예제파일/sample_docs.csv',
    'buyer_inquiry_email.txt': '5일차_실습자료_학생배포용/5일차_실습자료/01_규정문서/바이어문의메일_예시.txt',
    'alert_template.txt': '5일차_실습자료_학생배포용/5일차_실습자료/02_대시보드_데이터/알림문구_템플릿_예시.txt',
}

for target_name, rel_path in explicit_sources.items():
    src = os.path.join(root, rel_path.replace('/', os.sep))
    dst = os.path.join(data_dir, target_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied: {target_name}")
    else:
        print(f"File not found: {src}")

# Copy guideline text files
guidelines = [
    ('cbam_guideline_steel_알루미늄_실습용.txt', '학생용 실습예제파일/1일차 학생용 실습예제파일/cbam_guideline_steel_알루미늄_실습용.txt'),
    ('cbam_guideline_cement_fertilizer_실습용.txt', '학생용 실습예제파일/1일차 학생용 실습예제파일/cbam_guideline_cement_fertilizer_실습용.txt'),
    ('cbam_guideline_hydrogen_electricity_실습용.txt', '학생용 실습예제파일/1일차 학생용 실습예제파일/cbam_guideline_hydrogen_electricity_실습용.txt'),
    ('cbam_guideline_general_scope3_실습용.txt', '학생용 실습예제파일/1일차 학생용 실습예제파일/cbam_guideline_general_scope3_실습용.txt'),
]

knowledge = []
for fname, rel_path in guidelines:
    src = os.path.join(root, rel_path.replace('/', os.sep))
    if os.path.exists(src):
        dst = os.path.join(data_dir, fname)
        shutil.copy2(src, dst)
        with open(src, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        knowledge.append({
            "source": fname,
            "title": fname.replace('_실습용.txt', '').replace('cbam_guideline_', 'CBAM: '),
            "content": content
        })

# Parse PDFs using pypdf
try:
    from pypdf import PdfReader
    pdf1 = os.path.join(root, '5일차_실습자료_학생배포용/5일차_실습자료/01_규정문서/통관규정집_교육용_v1.pdf'.replace('/', os.sep))
    if os.path.exists(pdf1):
        reader = PdfReader(pdf1)
        text = "\n".join([f"[통관규정집 {i+1}p]\n" + (p.extract_text() or "") for i, p in enumerate(reader.pages)])
        knowledge.append({
            "source": "통관규정집_교육용_v1.pdf",
            "title": "수출입 통관 및 CBAM 규정집 (교육용)",
            "content": text
        })
        with open(os.path.join(data_dir, 'customs_regulation.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

    pdf2 = os.path.join(root, '5일차_실습자료_학생배포용/5일차_실습자료/01_규정문서/ESG공시가이드라인_교육용.pdf'.replace('/', os.sep))
    if os.path.exists(pdf2):
        reader = PdfReader(pdf2)
        text = "\n".join([f"[ESG공시가이드라인 {i+1}p]\n" + (p.extract_text() or "") for i, p in enumerate(reader.pages)])
        knowledge.append({
            "source": "ESG공시가이드라인_교육용.pdf",
            "title": "ESG 공시 및 공급망 실사 가이드라인 (교육용)",
            "content": text
        })
        with open(os.path.join(data_dir, 'esg_disclosure_guide.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
except Exception as e:
    print("PDF parse error:", e)

# Parse contract
try:
    import docx
    contract_p = os.path.join(root, '5일차_실습자료_학생배포용/5일차_실습자료/01_규정문서/신규수출계약서_발췌본.docx'.replace('/', os.sep))
    if os.path.exists(contract_p):
        doc = docx.Document(contract_p)
        text = "\n".join([p.text for p in doc.paragraphs if p.text])
        with open(os.path.join(data_dir, 'sample_contract.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        knowledge.append({
            "source": "신규수출계약서_발췌본.docx",
            "title": "신규 수출 물품 매매계약서 (Nordic Metals)",
            "content": text
        })
except Exception as e:
    print("Docx parse error:", e)

with open(os.path.join(data_dir, 'knowledge.json'), 'w', encoding='utf-8') as f:
    json.dump(knowledge, f, ensure_ascii=False, indent=2)

print("Finished setting up data directory successfully!")
