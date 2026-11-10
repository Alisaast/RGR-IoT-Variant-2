"""РГР, варіант 2: IoT теплиця / Edge Computing.
Запуск: python greenhouse_edge.py
Залежності: numpy pandas matplotlib (seaborn не потрібен).
Firebase: необов'язкова реальна інтеграція через service-account JSON та FIREBASE_DATABASE_URL.
"""
from pathlib import Path
import os, json, sqlite3, gzip, zlib, bz2, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(os.environ.get('RGR_OUTPUT_DIR', Path(__file__).resolve().parent / 'results'))
OUT.mkdir(parents=True, exist_ok=True)
RNG=np.random.default_rng(2026)
SITES=['Ділянка A','Ділянка B','Ділянка C']
DATES=pd.date_range('2025-01-01',periods=365,freq='D')

def simulate():
    records=[]
    for site_i,site in enumerate(SITES):
        for k,dt in enumerate(DATES):
            phase=2*np.pi*(k-80)/365
            temp=18+10*np.sin(phase)+RNG.normal(0,2.3)+site_i*.65
            moisture=52-6*np.sin(phase)+RNG.normal(0,5)+site_i*3
            lux=max(0,12500+8000*np.sin(phase)+RNG.normal(0,2300)-site_i*650)
            ph=6.35 + site_i*.18+RNG.normal(0,.19)
            readings=[moisture,temp,lux,ph]
            if RNG.random()<.024: readings[RNG.integers(0,4)]=None
            if RNG.random()<.014:
                j=RNG.integers(0,4)
                readings[j]=[135,-42,140000,11.4][j]
            records.append([dt.strftime('%Y-%m-%d'),site,*readings])
    df=pd.DataFrame(records,columns=['date','site','soil_moisture_pct','temperature_c','illuminance_lux','soil_ph'])
    df.to_csv(OUT/'raw.csv',index=False)
    return df

def clean(df):
    df=df.copy().sort_values(['site','date']).reset_index(drop=True)
    limits={'soil_moisture_pct':(0,100),'temperature_c':(-10,55),'illuminance_lux':(0,90000),'soil_ph':(3,9)}
    bad_counts={}
    for c,(lo,hi) in limits.items():
        bad=df[c].isna() | ~df[c].between(lo,hi)
        bad_counts[c]=int(bad.sum())
        df.loc[~df[c].between(lo,hi),c]=np.nan
        df[c]=df.groupby('site')[c].transform(lambda s: s.interpolate(limit_direction='both'))
        df[c]=df[c].fillna(df[c].median())
        df[c]=df[c].round(3)
    df['season']=pd.to_datetime(df.date).dt.month.map({12:'Зима',1:'Зима',2:'Зима',3:'Весна',4:'Весна',5:'Весна',6:'Літо',7:'Літо',8:'Літо',9:'Осінь',10:'Осінь',11:'Осінь'})
    df['ph_optimal']=df.soil_ph.between(6.0,7.0)
    df.to_csv(OUT/'cleaned.csv',index=False)
    (OUT/'quality.json').write_text(json.dumps({'rows':len(df),'invalid_or_missing':bad_counts},ensure_ascii=False,indent=2),encoding='utf8')
    return df

def database(df):
    path=OUT/'greenhouse.sqlite'
    if path.exists():path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript('''PRAGMA foreign_keys=ON;
        CREATE TABLE sites(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
        CREATE TABLE measurements(id INTEGER PRIMARY KEY, site_id INTEGER NOT NULL,
        date TEXT NOT NULL, season TEXT NOT NULL, moisture REAL NOT NULL,
        temperature REAL NOT NULL, lux REAL NOT NULL, ph REAL NOT NULL,
        optimal_ph INTEGER NOT NULL, FOREIGN KEY(site_id) REFERENCES sites(id),
        UNIQUE(site_id,date));''')
        conn.executemany('INSERT INTO sites(id,name) VALUES(?,?)',[(i+1,s) for i,s in enumerate(SITES)])
        rows=[(SITES.index(r.site)+1,r.date,r.season,r.soil_moisture_pct,r.temperature_c,r.illuminance_lux,r.soil_ph,int(r.ph_optimal)) for r in df.itertuples(index=False)]
        conn.executemany('INSERT INTO measurements(site_id,date,season,moisture,temperature,lux,ph,optimal_ph) VALUES(?,?,?,?,?,?,?,?)',rows)
        n=conn.execute('SELECT count(*) FROM measurements').fetchone()[0]
    return path,n

def plots(df):
    plt.rcParams['font.family']='DejaVu Sans'
    seasons=['Зима','Весна','Літо','Осінь']
    fig,ax=plt.subplots(figsize=(8,4.4))
    means=df.groupby('season')['temperature_c'].mean().reindex(seasons)
    ax.plot(seasons,means.values,marker='o');ax.set_ylabel('Температура, °C');ax.set_title('Сезонна динаміка температури');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'seasonal_temperature.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4))
    means=df.groupby('site')[['soil_moisture_pct','soil_ph']].mean()
    ax.bar(means.index,means.soil_moisture_pct);ax.set_ylabel('Середня вологість ґрунту, %');ax.set_title('Порівняння ділянок');fig.tight_layout();fig.savefig(OUT/'site_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4))
    ax.hist(df.soil_ph,bins=28);ax.axvline(6,ls='--',label='pH 6,0');ax.axvline(7,ls='--',label='pH 7,0');ax.set_xlabel('pH');ax.set_ylabel('Кількість вимірювань');ax.set_title('Розподіл pH (умовний робочий діапазон 6,0-7,0)');ax.legend();fig.tight_layout();fig.savefig(OUT/'ph_histogram.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4))
    ax.scatter(df.temperature_c,df.soil_moisture_pct,alpha=.22,s=10);ax.set_xlabel('Температура, °C');ax.set_ylabel('Вологість ґрунту, %');ax.set_title('Зв’язок температури та вологості');fig.tight_layout();fig.savefig(OUT/'temperature_moisture.png',dpi=150);plt.close(fig)

def compress(df):
    payload=df.to_json(orient='records',force_ascii=False).encode('utf8')
    compressed={'gzip':gzip.compress(payload,compresslevel=9),'zlib':zlib.compress(payload,level=9),'bzip2':bz2.compress(payload,compresslevel=9)}
    for name,blob in compressed.items():
        (OUT/('cleaned.json.'+{'gzip':'gz','zlib':'zlib','bzip2':'bz2'}[name])).write_bytes(blob)
    best=min(compressed,key=lambda n:len(compressed[n]))
    summary={name:{'bytes':len(blob),'ratio':round(len(payload)/len(blob),3),'saving_pct':round(100*(1-len(blob)/len(payload)),2)} for name,blob in compressed.items()}
    (OUT/'compression.json').write_text(json.dumps({'original_bytes':len(payload),'algorithms':summary,'best':best},ensure_ascii=False,indent=2),encoding='utf8')
    return best,summary

def report(df,n,best,summary):
    corr=df[['temperature_c','soil_moisture_pct']].corr().iloc[0,1]
    seasons=df.groupby('season')[['temperature_c','soil_moisture_pct','illuminance_lux','soil_ph']].mean().round(2)
    sites=df.groupby('site')[['temperature_c','soil_moisture_pct','illuminance_lux','soil_ph']].mean().round(2)
    optimal=round(df.ph_optimal.mean()*100,2)
    lines=['РГР - варіант 2. Звіт з моделювання теплиці',f'Записів у SQLite: {n}',f'Частка pH в умовному діапазоні 6,0-7,0: {optimal} %',f'Кореляція температура / вологість ґрунту: {corr:.3f}',f'Найкраще стиснення: {best}', '\nСезонні середні:',seasons.to_string(),'\nПорівняння ділянок:',sites.to_string(),'\nСтиснення:',json.dumps(summary,ensure_ascii=False,indent=2),'\nПРИМІТКА: усі показники синтетичні, а pH 6,0-7,0 - демонстраційна уставка, не універсальна агрономічна норма.']
    (OUT/'report.txt').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines))

def firebase_upload(df):
    """Тільки при явному налаштуванні облікових даних виконує справжній запис."""
    key=os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
    url=os.getenv('FIREBASE_DATABASE_URL')
    if not(key and url):
        (OUT/'firebase_status.txt').write_text('Firebase не налаштовано: немає FIREBASE_SERVICE_ACCOUNT_JSON та FIREBASE_DATABASE_URL. Локальний експорт готовий; запис у хмару НЕ виконано.',encoding='utf8')
        return False
    import firebase_admin
    from firebase_admin import credentials,db
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(key),{'databaseURL':url})
    records=json.loads(df.to_json(orient='records',force_ascii=False))
    db.reference('/rgr_variant_2/measurements').set(records)
    downloaded=db.reference('/rgr_variant_2/measurements').get()
    (OUT/'firebase_download.json').write_text(json.dumps(downloaded,ensure_ascii=False),encoding='utf8')
    (OUT/'firebase_status.txt').write_text(f'Успішна передача і повторне отримання {len(downloaded)} записів.',encoding='utf8')
    return True

if __name__=='__main__':
    raw=simulate();cleaned=clean(raw);dbpath,n=database(cleaned);plots(cleaned);best,summary=compress(cleaned);report(cleaned,n,best,summary)
    (OUT/'firebase_upload_payload.json').write_text(cleaned.to_json(orient='records',force_ascii=False),encoding='utf8')
    firebase_upload(cleaned)
    print(f'Результати: {OUT}')
