# ระบบ Dynamic Pricing โรงแรมแบบ 2 Agent

โปรเจกต์นี้จำลองลูกค้าที่เข้ามาค้นหา จอง และยกเลิกห้องพัก แล้วเปรียบเทียบ Fixed, Rule-based และ reinforcement-learning agent ที่ตั้งราคาใหม่ทุกชั่วโมง เป้าหมายคือกำไรสุทธิภายใต้ต้นทุน ค่าคอมมิชชัน การคืนเงิน ข้อจำกัดราคา และจำนวนห้อง

## ติดตั้ง

ต้องใช้ Python 3.11 ขึ้นไป

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## เตรียมข้อมูล

ไฟล์ดิบอยู่ใน `data/` และถูก gitignore

```powershell
kaggle datasets download -d jessemostipak/hotel-booking-demand --unzip -p data
```

ตรวจให้ไฟล์ชื่อ `data/hotel_bookings.csv` ก่อนรัน pipeline. หากมีสิทธิ์เข้าร่วมการแข่งขัน Expedia และยอมรับกติกาบน Kaggle แล้ว จึงดาวน์โหลด `train.csv` และเปลี่ยนชื่อเป็น `data/expedia_train.csv`. ถ้ายังไม่มีไฟล์ Expedia ระบบใช้ค่า default ที่ผู้ใช้กำหนดและบันทึกข้อจำกัดไว้ใน [ASSUMPTIONS.md](ASSUMPTIONS.md)

```powershell
kaggle competitions download -c expedia-personalized-sort -p data
Expand-Archive data\expedia-personalized-sort.zip -DestinationPath data\expedia
Move-Item data\expedia\train.csv data\expedia_train.csv
```

## เตรียมข้อมูลและ calibrate

```powershell
python -m src.data_prep
python -m src.tiering
python -m src.calibrate
python -m src.hotels
```

เปิด `notebooks/01_eda.ipynb` เพื่อดูกราฟสำรวจข้อมูล และดูไฟล์ผล EDA ที่ `results/eda/`. พารามิเตอร์การจำลอง, RL, evaluation และเส้นทางไฟล์อยู่ใน `config/sim_config.yaml`; `config/tier_profiles.json` เก็บผลที่ calibrate และป้ายที่มาของค่า

## ฝึกและประเมิน

ฝึกแต่ละ policy table แยกกัน:

```powershell
python -m src.train --algorithm n_step --mode domain_randomization
python -m src.train --algorithm n_step --mode single
python -m src.train --algorithm monte_carlo --mode domain_randomization
python -m src.train --algorithm monte_carlo --mode single
```

หรือใช้ `python -m src.train --suite` เพื่อรันทั้ง 4 แบบตามจำนวน episode ใน config. n-step domain-randomization บันทึกตารางหลักที่ `models/q_table.npy`; ตารางอื่นบันทึกชื่อแยกตาม algorithm และ mode ใน `models/`.

```powershell
python -m src.evaluate --models --stress --sensitivity --heatmap --replay
python -m src.plots
```

การประเมินหลักใช้โรงแรม test ที่ไม่อยู่ในชุดฝึก 10 แห่งต่อ tier และ 500 episode/tier; ทุก policy ใน episode เดียวกันได้ customer scenario เดียวกัน ไฟล์ CSV และกราฟอยู่ใน `results/`. ตารางแยกฤดูและ weekday/weekend, stress 5 โลก, sensitivity, policy heatmap และ hourly replay สร้างจากคำสั่งข้างต้น

## Demo และตรวจรับ

```powershell
streamlit run app.py
python -m pytest -q
```

Streamlit ให้เลือก tier, season, weekday/weekend, โมเดลและโรงแรมทดสอบ ดู KPI กับ heatmap และเรียก replay พร้อม customer log หรือ stress test ได้

## สรุปผลและข้อจำกัด

ผล full evaluation ใช้ 500 episode/tier/model บน test hotels ที่ไม่อยู่ใน train; ตารางแสดงกำไรสุทธิของ RL เทียบ Fixed:

| Model | Budget | Midscale | Luxury |
|---|---:|---:|---:|
| n-step DR | +2.91% | +21.18% | +22.54% |
| n-step single | +2.41% | +17.57% | +22.49% |
| Monte Carlo DR | +0.86% | +21.03% | +22.29% |
| Monte Carlo single | +1.52% | +20.24% | +22.48% |
| Rule-based | +5.41% | +6.76% | +3.53% |

ข้อเสนอจากผลนี้คือใช้ Rule-based กับ Budget และ RL กับ Midscale/Luxury; n-step DR เป็น policy หลักของ demo.

**อัปเดต 2026-11-03 · ตารางสุดท้าย** `models/q_table_final.npy` (คัดลอกเป็น `models/q_table.npy` ด้วย; ตารางเดิมยังอยู่ที่ `models/q_table_n_step_domain_randomization.npy`) เหมือน n-step DR ทุกค่าใน Budget/Midscale แต่ Luxury ฝึกใหม่โดยมีค่าปรับเมื่อ occupancy ต่ำกว่า Fixed − 3 จุด ผล test (`results/final/`, 500 episode/tier):

| Tier | ใช้ | กำไรเทียบ Fixed | Occupancy | Conversion |
|---|---|---:|---:|---:|
| Budget | Rule-based | +5.41% | 78.0% | 50.9% |
| Midscale | RL | +21.19% | 88.9% | 48.8% |
| Luxury | RL (ตารางสุดท้าย) | +20.79% | 91.0% | 51.3% |

Luxury ขายห้องได้ผ่านเกณฑ์แล้ว (เดิม 88.4%) โดยกำไรลดจาก +22.54% วิธีทดลอง เกณฑ์ และข้อจำกัดอยู่ใน [PLAN.md](PLAN.md) และ [ASSUMPTIONS.md](ASSUMPTIONS.md) Phase 7. หมายเหตุ: `python -m src.train` แบบ n-step DR จะบันทึกทับ `models/q_table.npy` ด้วยตารางที่ไม่มีค่าปรับ Stress test ยืนยันว่า Rule-based ชนะ Budget ทั้ง 5 โลก, RL ชนะ Rule-based ใน Midscale 4/5 โลก (แพ้เล็กน้อยเมื่อ demand ต่ำ) และชนะทุกโลกใน Luxury. Sensitivity ต่อ conversion rate และ hotel spread ให้ทิศทางเดียวกัน: Rule-based เหมาะกับ Budget ส่วน RL ยังดีกว่า baseline ใน Midscale/Luxury.

กราฟและรายละเอียดอยู่ใน `results/phase6_main_kpis.csv`, `phase6_stress_kpis.csv`, `phase6_sensitivity_kpis.csv`, ไฟล์ heatmap ต่อ model และ `phase6_replay_*.csv`. ทุก policy ใช้ customer scenario เดียวกัน; ไม่มีการปรับ simulator เพื่อให้ RL ชนะ. ผลอ้างอิงในโจทย์เป็นค่าคร่าว ๆ: ผลที่นี่ต่างเล็กน้อย โดยเฉพาะ Rule-based ของ Midscale (+6.76% เทียบช่วง +1–5%) และ n-step DR ของ Midscale (+21.18% เทียบช่วง +15–18%). ได้ตรวจ mask, reward/cancellation timing, common seeds, KPI denominator และ test suite แล้ว; Expedia ที่เข้าถึงไม่ได้และการ calibrate HBD เป็นข้อจำกัดที่อาจมีผล แต่ไม่สามารถระบุเหตุเชิงสาเหตุจากการทดลองนี้ได้

Hotel Booking Demand มีข้อมูลเฉพาะ booking ไม่ใช่ผู้เข้าชมทั้งหมด จึงประมาณ visitor count ด้วย conversion rate 0.35 และทดสอบ 0.25/0.45. Expedia ใช้ไม่ได้หากไม่มีบัญชีที่ยอมรับกติกา ทำให้ราคา tier และ hotel spread ยังเป็น default. การประมาณ β ใช้ room-type ADR เป็น proxy เพราะไม่มีขนาดห้องจริง ดูตัวเลขและข้อจำกัดทั้งหมดใน [ASSUMPTIONS.md](ASSUMPTIONS.md)
