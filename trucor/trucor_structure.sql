CREATE TABLE IF NOT EXISTS users(
					account_id INTEGER PRIMARY KEY,
					last_name TEXT,
					first_name TEXT,
					username TEXT,
					email TEXT,
					UNIQUE (email)
					);

CREATE TABLE IF NOT EXISTS ekgtemplate(
						created_at DATE,
						associated_account INTEGER,
						RQ REAL, 
						RS REAL, 
						RP REAL, 
						RT REAL,
						ST REAL,
						PQ REAL,
						PT REAL,
						RL REAL,
						LQ REAL,
						STPrime REAL,
						RTPrime REAL
						);
CREATE TABLE IF NOT EXISTS mitsample(
						created_at DATE,
						associated_account INTEGER,
						RQ REAL, 
						RS REAL, 
						RP REAL, 
						RT REAL,
						ST REAL,
						PQ REAL,
						PT REAL,
						RL REAL,
						LQ REAL,
						STPrime REAL,
						RTPrime REAL
						);
