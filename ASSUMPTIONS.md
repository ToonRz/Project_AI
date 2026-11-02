# สมมติฐานและที่มาของข้อมูล

## Phase 0 · ข้อมูลและการตั้งค่า

- ดาวน์โหลด Hotel Booking Demand จาก public GitHub mirror เพราะเครื่องนี้ไม่มี Kaggle API credentials; เก็บไฟล์ไว้ใน `data/` ซึ่ง `.gitignore` ป้องกันไม่ให้ commit ไฟล์ดิบ ชุดข้อมูลต้นฉบับเป็นของ Antonio, de Almeida และ Nunes (2019) ดู [แหล่ง mirror](https://github.com/chaithanya-sunil/Hotel-booking-eda) และ [บทความต้นฉบับ](https://doi.org/10.1016/j.dib.2018.11.126)
- ไม่มีไฟล์ Expedia Personalize Hotel Search ในเครื่อง หน้าแข่งขัน Kaggle ต้องให้ผู้ใช้ล็อกอินและยอมรับกติกาก่อน จึงไม่ได้ประมาณราคาอ้างอิงจาก Expedia, logistic price response หรือ spread รายโรงแรม ใช้ค่า default จากโจทย์และติดป้ายใน `config/tier_profiles.json`
- conversion rate ใช้ 0.35 เป็นค่าหลัก และทดสอบ sensitivity ที่ 0.25 และ 0.45
- โปรไฟล์ชั่วโมงที่ลูกค้าเข้ามา ขนาด/จำนวนห้อง ต้นทุนผันแปร ค่าคอมมิชชัน OTA และเวลายกเลิก ใช้ค่า default ที่ผู้ใช้ให้ เพราะข้อมูล Hotel Booking Demand ไม่มีฟิลด์เหล่านี้
- ไม่แปลงค่าเงินของ ADR ใน Hotel Booking Demand หรือราคา USD ใน Expedia เป็น THB เพราะไม่มีอัตราแลกเปลี่ยนและการจับคู่โรงแรมที่รองรับการแปลงอย่างมีหลักฐาน

## Phase 1 · ผลสำรวจข้อมูล

- อ่านข้อมูลดิบได้ 119,390 แถว 32 คอลัมน์
- หลังตัด `adr <= 0` และ ADR ที่สูงกว่า percentile 99.5 เหลือ 116,842 booking; cutoff เท่ากับ 275.21375 ในหน่วยเงินเดิมของชุดข้อมูล
- ขยายคืนเข้าพักเฉพาะ booking ที่ไม่ยกเลิกและมีคืนเข้าพักอย่างน้อยหนึ่งคืน ได้ 250,759 room-nights; ยังเก็บ booking ที่ยกเลิกไว้เพื่อวิเคราะห์ cancellation
- Leisure มี 111,400 booking (95.34%), lead time เฉลี่ย/มัธยฐาน 109.34/76 วัน, ADR มัธยฐาน 96, cancellation 38.39%, TA/TO 86.34%
- Business มี 5,442 booking (4.66%), lead time เฉลี่ย/มัธยฐาน 21.55/7 วัน, ADR มัธยฐาน 67, cancellation 19.06%, TA/TO 6.30%

## Phase 2 · การจัด tier และข้อจำกัดการประมาณ

- ไม่สามารถ fit ราคา tier, logistic booking-vs-price หรือความแปรผันระหว่าง Expedia properties ได้ จึงใช้ราคาอ้างอิงและ spread ตามโจทย์
- Hotel Booking Demand ไม่มี star rating, ขนาดห้อง หรือ mapping ของ `reserved_room_type` ไปยัง m² จึงประมาณ β โดยเลือก room code ที่พบบ่อยที่สุด 3 แบบ (A, D, E) เรียงตาม ADR มัธยฐานในแต่ละ segment แล้วจับคู่กับขนาดห้องสมมติจากเล็กไปใหญ่ วิธีนี้เป็น proxy ไม่ใช่ค่าจากขนาดห้องจริง
- β ที่ได้ (Leisure/Business): Budget 0.694/1.085, Midscale 0.679/1.060, Luxury 0.590/0.920 รายละเอียด room code และ ADR ที่ใช้บันทึกไว้ใน `config/tier_profiles.json`
- กำหนด Expedia rating 1–3 เป็น Budget, 4 เป็น Midscale และ 5 เป็น Luxury; ตัด rating 0 ซึ่งไม่มี tier ที่ระบุ

## Phase 3 · การ calibrate customer agent

- arrival weight 30 วันเป็น histogram ของ lead time จากข้อมูลจริงที่ normalize แล้ว; lead time 0 และ 1 นับเป็นวันแรก ส่วน booking ที่จองล่วงหน้าเกิน 30 วันอยู่นอก episode
- ประมาณผู้เข้าชมด้วย booking ÷ 0.35 ตาม conversion rate ตั้งต้น แต่ข้อมูลไม่มีจำนวนห้อง/กำลังรองรับ จึงประเมิน visitors ต่อ available room โดยตรงไม่ได้ และคง `demand_per_room_episode` ตามโจทย์
- leave-one-hotel-out สลับ City Hotel กับ Resort Hotel: ความคลาดเคลื่อน cumulative share สูงสุด 14.03% สำหรับ Leisure และ 12.86% สำหรับ Business; MAE รายวันสูงสุด 1.22 จุดเปอร์เซ็นต์
- log-ADR fit ได้ Leisure μ=4.5503, σ=0.4527 (111,400 แถว) และ Business μ=4.1791, σ=0.4509 (5,442 แถว) ใช้ σ ที่ calibrate ใน simulator แต่คง WTP median multiplier ตามโจทย์ เนื่องจากข้อมูลไม่มี star tier และไม่สามารถแปลงสกุลเงินอย่างมีหลักฐาน
- logistic cancellation fit ใช้ lead time และ deposit type ภายใน 30 วัน; ROC AUC ในชุด fit เท่ากับ 0.702 สำหรับ Leisure และ 0.656 สำหรับ Business จากนั้น project อัตราที่ marginalize ตาม deposit ลงบนตัวคูณ lead-time ตามสเปก และ scale ข้าม tier เพื่อรักษาลำดับค่า cancel ที่โจทย์กำหนด
- TA/TO share จากข้อมูลแทนค่า fallback: Leisure 86.34%, Business 6.30%. ตัวคูณ season: Leisure Low/Shoulder/High = 0.973/1.098/0.888; Business = 1.244/0.944/0.532. Weekday/weekend: Leisure 0.946/1.136; Business 1.186/0.533
- สัดส่วน Business รวมใน HBD เท่ากับ 4.66% แต่แยกต่อ tier จาก City/Resort ไม่ได้ จึงคง business share ต่อ tier ตามโจทย์
- สร้างโรงแรมจำลอง train 30 และ test 10 แห่งต่อ tier ด้วย seed 777; ID ของ train/test ไม่ซ้ำกัน

## Phase 4 · ผล baseline smoke check

Fixed และ Rule-based รันบนโรงแรม test ที่ไม่เคยเห็น 10 แห่งต่อ tier โดยใช้ seed ร่วมกัน ค่าเฉลี่ย net profit (THB):

| Tier | Fixed | Rule-based | Rule-based เทียบ Fixed |
|---|---:|---:|---:|
| Budget | 35,289.21 | 36,651.41 | +3.86% |
| Midscale | 96,611.44 | 104,477.11 | +8.14% |
| Luxury | 135,362.90 | 144,510.84 | +6.76% |

เป็น smoke check 10 episode/tier ไม่ใช่ผลประเมินหลัก 500 episode/tier; รายละเอียดอยู่ใน `results/baseline_kpis.csv` และ `results/baseline_episodes.csv`

## Phase 5 · Reinforcement learning

- ใช้ scenario seed เดียวกันข้าม policy, greedy evaluation ใช้ action mask และ n-step bootstrap ใช้เฉพาะ action ที่อนุญาตใน state ถัดไป
- ประเมิน state ที่ไม่เคยพบด้วยระดับราคา ×1.00 แล้ว clamp เข้า action mask ตามโจทย์; อ่าน visit-count table ที่ฝึกพร้อม Q-table เพื่อแยก state ที่ไม่เคยพบ
- พบและแก้การผูก cancellation reward: cancellation ที่มาถึงต้นชั่วโมงถูกบวกเข้ากับ transition ของ action ก่อนหน้า ก่อนอัปเดต learner ตามลำดับในโจทย์ ส่วน booking reward และ cancellation ที่เกิดทันทีในชั่วโมงปัจจุบันอยู่กับ transition ปัจจุบัน
- smoke evaluation 10 episode/tier ของทั้งสี่ตาราง RL เทียบ Fixed ได้:

| Model | Budget | Midscale | Luxury |
|---|---:|---:|---:|
| n-step DR | −0.46% | +28.92% | +21.01% |
| n-step single | −0.30% | +18.82% | +27.60% |
| Monte Carlo DR | −2.32% | +29.20% | +19.78% |
| Monte Carlo single | −0.58% | +22.93% | +20.52% |

Rule-based เทียบ Fixed ได้ Budget +3.86%, Midscale +8.14%, Luxury +6.76%. RL ผ่าน RL ≥ Fixed ใน Midscale/Luxury ทุกแบบ แต่ Budget Rule-based ยังดีกว่า ผลนี้เป็น acceptance check 10 episode/tier ไม่ใช่ผลหลัก 500 episode/tier. ตารางเต็มอยู่ใน `results/phase5_model_comparison.csv`
- บันทึก learning curves, Q-table และ update-count tables ของทุก variant ไว้ใน `results/` และ `models/`; ไม่ใช้ผลจากตารางที่ฝึกก่อนแก้ reward timing เป็นผลสรุป

## Phase 6 · วิธีประเมินและข้อสมมติของ sensitivity

- ผลหลักวนโรงแรม test 10 แห่งต่อ tier ให้ครบ 500 episode/tier โดยสร้าง season, weekend และ customer scenario จาก seed สูตรที่ตั้งใน config เดียวกันสำหรับทุก policy
- Stress test เปลี่ยนเฉพาะ WTP median, demand หรือ σ ตามโลกที่ระบุใน config; ไม่แก้ reward, ราคา, cost หรือ parameter อื่นของ simulator
- Conversion-rate sensitivity แปลง conversion ที่ต้องการเป็น visitor demand โดยคูณ `base_conversion_rate / target_conversion_rate`; ค่ากำลังรองรับยังเป็น default เพราะข้อมูลไม่มี capacity
- Spread sensitivity คูณ spread แบบ multiplicative และ additive รวมถึง hShift และ season/weekend amplitude spread ที่ใช้สร้างโรงแรมจำลอง ก่อนสร้าง test population ใหม่ด้วย seed เดิม; ค่าขอบเขตการ clamp ไม่เปลี่ยน
- Heatmap แสดง markup จาก representative state ของ day bucket, ช่วงเวลาหกชั่วโมง, season, weekday/weekend และ remaining bucket ที่เลือกใน config; เป็นการสรุป policy ไม่ใช่การสุ่ม replay
- Replay ใช้โรงแรม test ตาม index ใน config และ scenario seed เดียวกันสำหรับ Fixed, Rule-based และ RL
- Full main evaluation มี 18,000 แถว: 4 model × 3 tier × 3 policy × 500 episode. RL เทียบ Fixed: n-step DR Budget/Midscale/Luxury = +2.91/+21.18/+22.54%; n-step single = +2.41/+17.57/+22.49%; Monte Carlo DR = +0.86/+21.03/+22.29%; Monte Carlo single = +1.52/+20.24/+22.48%. Rule-based = +5.41/+6.76/+3.53% ตามลำดับ
- Stress มี 450 episode จาก 5 worlds × 3 tiers × 10 hotels × 3 policies; sensitivity มี 540 episode จาก 6 one-factor cases × 3 tiers × 10 hotels × 3 policies. Heatmap มี 2,160 state/model; replay CSV มี hourly 19,440 แถว, daily 237 แถว และ customer log 804 แถว
- Full-sample recommendation: Rule-based สำหรับ Budget (ชนะ RL ใน main และทุก stress world); RL สำหรับ Midscale และ Luxury. ใน stress, RL ชนะ Rule-based ใน Midscale 4/5 โลกและ Luxury 5/5 โลก
- ผลต่างเล็กน้อยจากช่วง lab reference: Rule-based Midscale +6.76% เทียบช่วง +1–5%; n-step DR Midscale +21.18% เทียบ +15–18%. ตรวจ action mask, cancellation/reward alignment, seeds, baseline reference และ KPI aggregation แล้ว โดยไม่ได้ปรับ simulator; Expedia unavailable และ HBD-based calibration เป็นข้อจำกัด แต่สรุปสาเหตุจากผลนี้โดยตรงไม่ได้
