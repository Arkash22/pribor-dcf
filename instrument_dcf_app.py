import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="DCF — Приборостроение", layout="wide")

st.title("DCF-модель для проекта по приборостроению")
st.caption(
    "Гибкая модель: план продаж по годам + склад (выпуск ≠ продажи). "
    "Год 1: Dev + РУ, без ФОТ/выручки. Год 2: CapEx и запуск производства. "
    "Отдельный NPV для лицензиара (получателя роялти). НДС добавлен как инфо-показатель. "
    "Добавлены накладные и маркетинговые расходы."
)

# --------- Sidebar Inputs ---------
st.sidebar.header("Входные параметры")

DevCost = st.sidebar.number_input("Стоимость разработки прототипа (₽)", min_value=0, value=10_000_000, step=100_000)
RU_cost = st.sidebar.number_input("Стоимость РУ (рег. удостоверения), ₽", min_value=0, value=5_000_000, step=100_000)
CapEx_infra = st.sidebar.number_input("Капвложения в инфраструктуру (разово, Год 2), ₽", min_value=0, value=5_000_000, step=100_000)

Salary = st.sidebar.number_input("ФОТ инженера с налогами (₽/мес.)", min_value=0, value=250_000, step=10_000)
N_eng = st.sidebar.slider("Количество инженеров (Год 2+)", min_value=1, max_value=5, value=3, step=1)
# Новый рычаг: накладные сразу после количества инженеров
OverheadPct = st.sidebar.slider("Накладные расходы, % от годового ФОТ", min_value=1, max_value=500, value=100, step=1) / 100.0

ProdPerEng = st.sidebar.number_input("Производительность 1 инженера (шт./мес.)", min_value=0.0, value=1.0, step=0.1, format="%.1f")

MatCost = st.sidebar.number_input("Материалы на изделие (₽/ед.)", min_value=0, value=600_000, step=10_000)
Price = st.sidebar.number_input("Цена продажи (рыночная), ₽/ед.", min_value=0, value=1_500_000, step=10_000)

RoyaltyPct = st.sidebar.slider("Лицензионное вознаграждение, % от выручки", min_value=0.0, max_value=50.0, value=5.0, step=0.5) / 100.0
# Новый рычаг: маркетинг сразу под роялти
MarketingPct = st.sidebar.slider("Маркетинговые расходы, % от выручки", min_value=1, max_value=30, value=5, step=1) / 100.0

DiscRate = st.sidebar.slider("Ставка дисконтирования (годовая), %", min_value=5.0, max_value=40.0, value=18.0, step=0.5) / 100.0
T = st.sidebar.slider("Горизонт планирования, лет", min_value=2, max_value=15, value=5, step=1)

TaxRate = 0.25          # Налог на прибыль 25%
VAT_RATE = 0.20         # НДС 20% на произведённую единицу (инфо-показатель)

st.sidebar.markdown("---")
st.sidebar.info(f"Налог на прибыль: **{int(TaxRate*100)}%**. НДС (инфо): **{int(VAT_RATE*100)}%** на произведённую единицу.")
st.sidebar.caption(
    "Год 1 — только Dev + РУ. Производство и ФОТ — с Года 2. "
    "Выпуск НЕ равен продажам; непроданное → остатки на складе. "
    "Если спрос опережает выпуск, продаём из остатков. "
    "Накладные считаются как % от годового ФОТ, маркетинг — как % от выручки."
)

# --------- Capacity & Production ---------
years = list(range(1, T+1))
annual_payroll = Salary * N_eng * 12
capacity_per_year = N_eng * ProdPerEng * 12  # шт./год

# План продаж (шт./год) — редактируемая таблица
auto_default_sales = [0.0 if y == 1 else capacity_per_year for y in years]
plan_df = pd.DataFrame({"Год": years, "План продаж, шт.": auto_default_sales})

st.subheader("План продаж (шт./год)")
edited_plan_df = st.data_editor(
    plan_df,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
)

# ПРОИЗВОДСТВО: производим = мощности (с Года 2), в Год 1 = 0
production_units = [0.0 if y == 1 else capacity_per_year for y in years]

# Фактические продажи с учётом складов: продаём min(план, выпуск + начальные остатки)
plan_sales = edited_plan_df["План продаж, шт."].astype(float).tolist()
sales_units = []
ending_inventory_units = []
inv_start = 0.0
for i, y in enumerate(years):
    available = inv_start + production_units[i]
    sell = min(max(0.0, plan_sales[i]), available)
    sales_units.append(sell)
    inv_end = available - sell
    ending_inventory_units.append(inv_end)
    inv_start = inv_end  # переход на следующий год

st.subheader("Мощность, выпуск и продажи")
cap_df = pd.DataFrame({
    "Год": years,
    "Мощность, шт./год": [capacity_per_year]*len(years),
    "Выпуск, шт.": production_units,
    "Продажи, шт.": sales_units,
    "Остатки на складе (конец года), шт.": ending_inventory_units,
    "Загрузка, % (по выпуску)": [0.0 if (capacity_per_year==0 or y==1) else 100.0*production_units[i]/capacity_per_year for i, y in enumerate(years)],
})
st.dataframe(cap_df, use_container_width=True)

# --------- Financials: Manufacturer (Производитель/Лицензиат) ---------
revenue = [Price * sales_units[i] for i, _ in enumerate(years)]
materials = [MatCost * production_units[i] for i, _ in enumerate(years)]
payroll = [0.0 if y == 1 else annual_payroll for y in years]
royalty = [revenue[i] * RoyaltyPct for i, _ in enumerate(years)]

# Накладные: % от годового ФОТ (начиная со 2-го года)
overhead_year = annual_payroll * OverheadPct
overhead = [0.0 if y == 1 else overhead_year for y in years]

# Маркетинг: % от выручки
marketing = [revenue[i] * MarketingPct for i, _ in enumerate(years)]

# НДС как инфо-показатель: 20% от цены * произведённые единицы
vat_amount = [VAT_RATE * Price * production_units[i] for i, _ in enumerate(years)]

# EBITDA (вычитаем материалы, ФОТ, роялти, накладные, маркетинг)
ebitda = [revenue[i] - materials[i] - payroll[i] - royalty[i] - overhead[i] - marketing[i] for i in range(len(years))]

# Налог на прибыль (упрощённо от положительной EBITDA) — 25%
tax_profit = [max(0.0, ebitda[i] * TaxRate) for i in range(len(years))]

# Чистая прибыль
net_income = [ebitda[i] - tax_profit[i] for i in range(len(years))]

# Инвестиции/разовые: Год 1 -> Dev+RU, Год 2 -> CapEx_infra
invest = [(DevCost + RU_cost) if y == 1 else (CapEx_infra if y == 2 else 0.0) for y in years]

# FCF (упрощённо)
fcf = [net_income[i] - invest[i] for i in range(len(years))]

# --------- Financials: Licensor (Лицензиар) ---------
lic_revenue = royalty[:]
lic_ebitda = lic_revenue  # нет опер. затрат в упрощении
lic_tax = [max(0.0, lic_ebitda[i] * TaxRate) for i in range(len(years))]
lic_net_income = [lic_ebitda[i] - lic_tax[i] for i in range(len(years))]
lic_invest = [0.0 for _ in years]
lic_fcf = [lic_net_income[i] - lic_invest[i] for i in range(len(years))]

# --------- Discounting / KPIs ---------
dfactors = [1.0 / ((1.0 + DiscRate) ** y) for y in years]

# Manufacturer
dcf = [fcf[i] * dfactors[i] for i in range(len(years))]
cum_fcf = np.cumsum(fcf).tolist()

# Licensor
lic_dcf = [lic_fcf[i] * dfactors[i] for i in range(len(years))]

def irr_bisection(cashflows, tol=1e-6, max_iter=200):
    def npv_indexed(rate):
        return sum(cf / ((1+rate)**(i+1)) for i, cf in enumerate(cashflows))
    low, high = -0.99, 10.0
    f_low = npv_indexed(low)
    f_high = npv_indexed(high)
    if f_low*f_high > 0:
        return float("nan")
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = npv_indexed(mid)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return mid

# KPIs
NPV_manu = float(np.sum(dcf))
IRR_manu = irr_bisection(fcf)

NPV_lic = float(np.sum(lic_dcf))
IRR_lic = irr_bisection(lic_fcf)

# Payback (только для производителя)
payback = None
for i, y in enumerate(years):
    if cum_fcf[i] >= 0:
        if i == 0:
            payback = 0.0
        else:
            prev_cum = cum_fcf[i-1]
            need = -prev_cum
            cf_year = fcf[i]
            frac = (need / cf_year) if cf_year != 0 else 0.0
            payback = (y-1) + frac
        break

# --------- Summary KPIs ---------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NPV (Производитель), ₽", f"{NPV_manu:,.0f}".replace(",", " "))
c2.metric("IRR (Производитель), %", f"{IRR_manu*100:,.1f}" if not math.isnan(IRR_manu) else "н/д")
c3.metric("Окупаемость, лет", f"{payback:,.2f}" if payback is not None else "> горизонта")
c4.metric("NPV (Лицензиар), ₽", f"{NPV_lic:,.0f}".replace(",", " "))
c5.metric("Мощность (шт./год)", f"{capacity_per_year:,.1f}".replace(",", " "))

st.markdown("---")

# --------- Financial tables ---------
fin_df = pd.DataFrame({
    "Год": years,
    "Выпуск, шт.": production_units,
    "Продажи, шт.": sales_units,
    "Остатки на складе, шт.": ending_inventory_units,
    "ФОТ, ₽": payroll,
    "Материалы, ₽": materials,
    "Роялти, ₽": royalty,
    "Накладные, ₽": overhead,
    "Маркетинг, ₽": marketing,
    "НДС, ₽": vat_amount,
    "Выручка, ₽": revenue,
    "EBITDA, ₽": ebitda,
    "Чистая прибыль, ₽": net_income,
    "Инвестиции, ₽": invest,
    "Накопленный FCF, ₽": cum_fcf,
})

lic_cum_fcf = np.cumsum(lic_fcf).tolist()
lic_df = pd.DataFrame({
    "Год": years,
    "Выручка от лицензии, ₽": lic_revenue,
    "EBITDA (лицензиар), ₽": lic_ebitda,
    "Чистая прибыль (лицензиар), ₽": lic_net_income,
    "Накопленный FCF (лицензиар), ₽": lic_cum_fcf,
})

st.subheader("Финансовые результаты — Производитель")
st.dataframe(
    fin_df.style.format({
        "ФОТ, ₽": "{:,.0f}".format,
        "Материалы, ₽": "{:,.0f}".format,
        "Роялти, ₽": "{:,.0f}".format,
        "Накладные, ₽": "{:,.0f}".format,
        "Маркетинг, ₽": "{:,.0f}".format,
        "НДС, ₽": "{:,.0f}".format,
        "Выручка, ₽": "{:,.0f}".format,
        "EBITDA, ₽": "{:,.0f}".format,
        "Чистая прибыль, ₽": "{:,.0f}".format,
        "Инвестиции, ₽": "{:,.0f}".format,
        "Накопленный FCF, ₽": "{:,.0f}".format,
    }).set_properties(**{"text-align": "right"}),
    use_container_width=True
)

st.subheader("Финансовые результаты — Лицензиар")
st.dataframe(
    lic_df.style.format({
        "Выручка от лицензии, ₽": "{:,.0f}".format,
        "EBITDA (лицензиар), ₽": "{:,.0f}".format,
        "Чистая прибыль (лицензиар), ₽": "{:,.0f}".format,
        "Накопленный FCF (лицензиар), ₽": "{:,.0f}".format,
    }).set_properties(**{"text-align": "right"}),
    use_container_width=True
)

# --------- Charts ---------
st.subheader("Графики")

# 1) FCF производителя
fig1, ax1 = plt.subplots()
ax1.plot(years, fcf, marker="o")
ax1.set_title("Свободный денежный поток (FCF) — Производитель")
ax1.set_xlabel("Год")
ax1.set_ylabel("₽")
st.pyplot(fig1)

# 2) DCF производителя
fig2, ax2 = plt.subplots()
ax2.bar(years, dcf)
ax2.set_title("Дисконтированные потоки (DCF) — Производитель")
ax2.set_xlabel("Год")
ax2.set_ylabel("₽")
st.pyplot(fig2)

# 3) P&L производителя: Выручка vs EBITDA vs Net Income
fig3, ax3 = plt.subplots()
ax3.plot(years, revenue, marker="o", label="Выручка")
ax3.plot(years, ebitda, marker="o", label="EBITDA")
ax3.plot(years, net_income, marker="o", label="Чистая прибыль")
ax3.set_title("P&L динамика — Производитель")
ax3.set_xlabel("Год")
ax3.set_ylabel("₽")
ax3.legend()
st.pyplot(fig3)

# 4) DCF лицензиара
fig4, ax4 = plt.subplots()
ax4.bar(years, lic_dcf)
ax4.set_title("Дисконтированные потоки (DCF) — Лицензиар")
ax4.set_xlabel("Год")
ax4.set_ylabel("₽")
st.pyplot(fig4)

st.markdown("---")
st.caption(
    "Производство = мощности (с Года 2). Если спрос > выпуска, продаём из остатков; непроданное идёт в конец-года остатки.\n"
    "НДС показан информативно (20% от цены * произведённые единицы) и в EBITDA не включён. Налог на прибыль — 25%.\n"
    "Роялти, накладные (от ФОТ) и маркетинг (от выручки) учитываются в операционных расходах производителя."
)
