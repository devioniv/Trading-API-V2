# TradingAPI_v12.py
# Full app: Select-All strategies + per-instrument/global data-type + robust reqAccountUpdatesAsync,
# candlestick confirmation logs (historical & live bars), and PySide6 enum compatibility fixes.
#
# Save as TradingAPI_v11.py and run locally (paper trading recommended).
# Requires: PySide6, qasync, pandas, pandas_ta (ib_async optional)

import sys
import asyncio
import os
import json
import time
import traceback
import inspect
from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QLabel, QLineEdit, QComboBox,
    QTextEdit, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QDoubleSpinBox, QAbstractItemView, QCheckBox
)
from PySide6.QtCore import Signal, Qt

# Try to import IB libs — if missing, UI still works in fake-data mode.
try:
    from ib_async import IB, util, Stock, Forex, Future, Option, Crypto, MarketOrder, LimitOrder
    from numpy.random import default_rng # For modern numpy random generation
except Exception:
    IB = None
    MarketOrder = None
    LimitOrder = None
    default_rng = None # Ensure it exists for type hinting

import pandas as pd
import pandas_ta as ta
import qasync

# ---- Constants ----
CONFIG_FILE = "trading_config.json"

# ---- Helpers ----
def normalize_ta_columns(df: pd.DataFrame):
    """Best-effort normalization for ATR/MACD column names from pandas_ta variations."""
    import re
    try:
        if df is None:
            return
        if "ATRr_14" in df.columns and "ATR_14" not in df.columns:
            df.rename(columns={"ATRr_14": "ATR_14"}, inplace=True)
        macd_candidates = [c for c in df.columns if re.search(r'(?i)macd', c)]
        main = signal = hist = None
        for c in macd_candidates:
            lc = c.lower()
            if lc == "macd" or (lc.startswith("macd_") and main is None):
                main = c
            if "signal" in lc and signal is None:
                signal = c
            if "hist" in lc and hist is None:
                hist = c
        renames = {}
        if main and main != "MACD":
            renames[main] = "MACD"
        if signal and signal != "MACD_signal":
            renames[signal] = "MACD_signal"
        if hist and hist != "MACD_hist":
            renames[hist] = "MACD_hist"
        if renames:
            df.rename(columns=renames, inplace=True)
    except Exception:
        pass

DATA_TYPE_MAP = {
    "Live": 1,
    "Frozen": 2,
    "Delayed": 3,
    "Delayed-Frozen": 4,
    "Historical": 5
}

# ---- Main Application ----
class TradingApp(QMainWindow):
    connection_status_signal = Signal(bool)
    account_update_signal = Signal(str, str, str)
    position_update_signal = Signal(str, float, float)
    instrument_list_update_signal = Signal(list)
    log_signal = Signal(str)
    summary_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TradingAPI v11")
        self.resize(1400, 920)

        # IB client (optional)
        self.ib = IB() if IB is not None else None

        # Defensive event binding
        if self.ib is not None:
            for event_name in ("connectedEvent", "disconnectedEvent", "orderStatusEvent",
                               "execDetailsEvent", "errorEvent", "accountValueEvent", "positionEvent"):
                try:
                    evt = getattr(self.ib, event_name)
                    # bind common handlers if event exists
                    if event_name == "connectedEvent":
                        evt += self.on_connected
                    elif event_name == "disconnectedEvent":
                        evt += self.on_disconnected
                    elif event_name == "orderStatusEvent":
                        evt += self.on_order_status
                    elif event_name == "execDetailsEvent":
                        evt += self.on_exec_details
                    elif event_name == "errorEvent":
                        evt += self.on_ib_error
                    elif event_name == "accountValueEvent":
                        evt += self.on_account_value_event
                    elif event_name == "positionEvent":
                        evt += self.on_position_event
                except Exception:
                    pass

        # state
        self.tasks = set()
        self.instruments: Dict[str, Dict[str, Any]] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self.strategies: Dict[int, object] = {}
        self.strategy_tasks: Dict[int, asyncio.Task] = {}
        self.next_strategy_id = 1

        # signals
        self.log_signal.connect(self.append_log)
        self.connection_status_signal.connect(self.update_connection_status)
        self.account_update_signal.connect(self.update_account_table)
        self.position_update_signal.connect(self.update_positions_table)
        self.instrument_list_update_signal.connect(self.update_instrument_selectors)
        self.summary_signal.connect(self.update_summary_text)

        # build UI
        self.create_widgets()
        self.load_config()

        # Attempt binding to live bar events if IB exposes them — defensive
        self._bind_live_bar_events()

    # UI building
    def create_widgets(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # Top row: Connect + global data type + instrument add
        top = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_ib)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_ib)
        top.addWidget(self.connect_btn)
        top.addWidget(self.disconnect_btn)

        top.addSpacing(12)
        top.addWidget(QLabel("Global Market Data Type:"))
        self.global_data_type = QComboBox()
        for name in DATA_TYPE_MAP:
            self.global_data_type.addItem(name, DATA_TYPE_MAP[name])
        top.addWidget(self.global_data_type)
        self.apply_global_dt_btn = QPushButton("Apply Global DataType")
        self.apply_global_dt_btn.clicked.connect(self.on_apply_global_data_type)
        top.addWidget(self.apply_global_dt_btn)

        top.addSpacing(12)
        top.addWidget(QLabel("Symbol:"))
        self.symbol_entry = QLineEdit()
        self.symbol_entry.setFixedWidth(120)
        top.addWidget(self.symbol_entry)
        top.addWidget(QLabel("SecType:"))
        self.sec_type = QComboBox()
        self.sec_type.addItems(["STK", "CASH", "FUT", "OPT", "CRYPTO"])
        top.addWidget(self.sec_type)
        self.add_inst_btn = QPushButton("Add Instrument")
        self.add_inst_btn.clicked.connect(self.on_add_instrument)
        top.addWidget(self.add_inst_btn)

        v.addLayout(top)

        # Tabs
        self.tabs = QTabWidget()
        v.addWidget(self.tabs)

        # Dashboard tab
        self.tab_dash = QWidget()
        self.tabs.addTab(self.tab_dash, "Dashboard")
        dash_layout = QHBoxLayout(self.tab_dash)

        left = QVBoxLayout()
        self.account_table = QTableWidget(0, 3)
        self.account_table.setHorizontalHeaderLabels(["Tag", "Value", "Currency"])
        self._set_header_resize(self.account_table, 3)
        left.addWidget(QLabel("Account"))
        left.addWidget(self.account_table)

        self.positions_table = QTableWidget(0, 3)
        self.positions_table.setHorizontalHeaderLabels(["Symbol", "Quantity", "Avg Cost"])
        self._set_header_resize(self.positions_table, 3)
        left.addWidget(QLabel("Positions"))
        left.addWidget(self.positions_table)

        dash_layout.addLayout(left, 2)
        right = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        right.addWidget(QLabel("Summary"))
        right.addWidget(self.summary_text)
        dash_layout.addLayout(right, 1)

        # Instruments tab
        self.tab_inst = QWidget()
        self.tabs.addTab(self.tab_inst, "Instruments")
        inst_layout = QVBoxLayout(self.tab_inst)

        self.inst_table = QTableWidget(0, 3)
        self.inst_table.setHorizontalHeaderLabels(["Symbol", "SecType", "DataType"])
        self._set_header_resize(self.inst_table, 3)
        # allow multi-row selection
        self.inst_table.setSelectionBehavior(getattr(QAbstractItemView, "SelectRows", QAbstractItemView.SelectItems))
        self.inst_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        inst_layout.addWidget(self.inst_table)

        inst_btns = QHBoxLayout()
        self.fetch_history_btn = QPushButton("Fetch History (selected rows)")
        self.fetch_history_btn.clicked.connect(self.on_fetch_history_selected)
        inst_btns.addWidget(self.fetch_history_btn)
        self.remove_inst_btn = QPushButton("Remove Selected")
        self.remove_inst_btn.clicked.connect(self.on_remove_selected_instruments)
        inst_btns.addWidget(self.remove_inst_btn)
        inst_layout.addLayout(inst_btns)

        # Strategy Manager tab
        self.tab_strat = QWidget()
        self.tabs.addTab(self.tab_strat, "Strategy Manager")
        strat_layout = QVBoxLayout(self.tab_strat)

        # Select-all checkbox + per-strategy checkboxes
        top_strat_row = QHBoxLayout()
        self.select_all_chk = QCheckBox("Select All Strategies")
        self.select_all_chk.stateChanged.connect(self.on_select_all_toggled)
        top_strat_row.addWidget(self.select_all_chk)
        top_strat_row.addStretch()
        strat_layout.addLayout(top_strat_row)

        self.strategy_types = ["WMA_SMMA_MACD", "TrendlineBreakout", "SupportResistance", "FairValueGap"]
        self.strategy_checkboxes: Dict[str, QCheckBox] = {}
        strat_box = QGroupBox("Available Strategies")
        strat_box_layout = QHBoxLayout()
        for s in self.strategy_types:
            cb = QCheckBox(s)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_individual_strategy_toggled)
            self.strategy_checkboxes[s] = cb
            strat_box_layout.addWidget(cb)
        strat_box.setLayout(strat_box_layout)
        strat_layout.addWidget(strat_box)

        # parameters row
        params_row = QHBoxLayout()
        self.apply_to_selected_chk = QCheckBox("Apply to selected instruments (rows)")
        self.apply_to_selected_chk.setChecked(True)
        params_row.addWidget(self.apply_to_selected_chk)
        params_row.addWidget(QLabel("Single Symbol:"))
        self.single_symbol_dropdown = QComboBox()
        params_row.addWidget(self.single_symbol_dropdown)
        params_row.addWidget(QLabel("Qty:"))
        self.qty_spin = QDoubleSpinBox()
        self.qty_spin.setValue(1.0)
        params_row.addWidget(self.qty_spin)
        params_row.addWidget(QLabel("Interval(s):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setValue(5.0)
        params_row.addWidget(self.interval_spin)
        self.auto_trade_chk = QCheckBox("Auto Trade (LIVE)")
        params_row.addWidget(self.auto_trade_chk)
        self.create_checked_btn = QPushButton("Create + Start Checked Strategies")
        self.create_checked_btn.clicked.connect(self.on_create_checked_strategies)
        params_row.addWidget(self.create_checked_btn)
        strat_layout.addLayout(params_row)

        # Running strategies table
        self.running_strats_table = QTableWidget(0, 6)
        self.running_strats_table.setHorizontalHeaderLabels(["ID", "Symbol", "Strategy", "Qty", "Interval", "Action"])
        self._set_header_resize(self.running_strats_table, 6)
        strat_layout.addWidget(self.running_strats_table)

        # Logs tab
        self.tab_logs = QWidget()
        self.tabs.addTab(self.tab_logs, "Logs")
        logs_layout = QVBoxLayout(self.tab_logs)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        logs_layout.addWidget(self.log_text)

        self.statusBar().showMessage("Ready")

    # safe header resize helper (handles PySide6/QHeaderView enum differences)
    def _set_header_resize(self, table: QTableWidget, cols: int):
        try:
            header = table.horizontalHeader()
            # Try both enum styles to avoid Pylance errors
            if hasattr(QHeaderView, "Stretch"):
                mode = QHeaderView.Stretch
            else:
                mode = getattr(QHeaderView, "ResizeMode", QHeaderView).Stretch
            for c in range(cols):
                header.setSectionResizeMode(c, mode)
        except Exception:
            try:
                # fallback single call
                table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            except Exception:
                pass

    # ---- core helpers ----
    def add_task(self, coro):
        try:
            if asyncio.iscoroutine(coro):
                task = asyncio.create_task(coro)
            elif callable(coro):
                maybe = coro()
                if asyncio.iscoroutine(maybe):
                    task = asyncio.create_task(maybe)
                else:
                    loop = asyncio.get_event_loop()
                    task = loop.run_in_executor(None, coro)
            else:
                self.log("add_task: expected coroutine or callable returning coroutine")
                return None
        except Exception as e:
            try:
                task = asyncio.create_task(coro)
            except Exception as e2:
                self.log(f"add_task error: {e} / {e2}")
                return None
        try:
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        except Exception:
            pass
        return task

    def append_log(self, msg: str):
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self.log_text.append(f"[{ts}] {msg}")
        except Exception:
            pass

    def log(self, msg: str):
        try:
            self.log_signal.emit(str(msg))
        except Exception:
            pass

    # ---- IB connection / events ----
    def connect_ib(self):
        self.log("Attempting IB connect...")
        if self.ib is None:
            self.log("IB library not available in environment.")
            return
        self.add_task(self._connect_async())

    async def _connect_async(self):
        try:
            await self.ib.connectAsync("127.0.0.1", 7497, clientId=1)
            self.log("IB connected (async).")
        except Exception as e:
            self.log(f"IB connect failed: {e}")

    def disconnect_ib(self):
        try:
            if self.ib is not None:
                self.ib.disconnect()
                self.log("IB disconnect requested.")
        except Exception as e:
            self.log(f"Disconnect error: {e}")

    def on_connected(self):
        # robustly call reqAccountUpdatesAsync with correct signature via inspect
        try:
            self.connection_status_signal.emit(True)
        except Exception:
            pass

        if not self.ib:
            return

        traces = []
        success = False
        try:
            if hasattr(self.ib, "reqAccountUpdatesAsync"):
                func = getattr(self.ib, "reqAccountUpdatesAsync")
                try:
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    # common possibilities: ("subscribe","acctCode") or ("acctCode",) or ("accountCode",)
                    if "subscribe" in params or "acctCode" in params or "accountCode" in params:
                        # try keyword call first
                        try:
                            if "subscribe" in params and "acctCode" in params:
                                res = func(subscribe=True, acctCode="")
                                traces.append("Called reqAccountUpdatesAsync(subscribe=True, acctCode='')")
                            elif "accountCode" in params:
                                res = func("")
                                traces.append("Called reqAccountUpdatesAsync('')  # accountCode signature")
                            else:
                                # fallback positional
                                res = func(True, "")
                                traces.append("Called reqAccountUpdatesAsync(True, '') fallback")
                        except TypeError as te:
                            traces.append(f"TypeError while calling reqAccountUpdatesAsync: {te}")
                            # fallback try positional
                            try:
                                res = func(True, "")
                                traces.append("Called reqAccountUpdatesAsync(True, '') fallback2")
                            except Exception as e:
                                traces.append(f"Second fallback exception: {e}")
                                res = None
                        if res is not None:
                            if asyncio.iscoroutine(res):
                                self.add_task(res)
                            success = True
                except Exception as e:
                    traces.append(f"inspect or call error: {e}")
            if not success and hasattr(self.ib, "reqAccountUpdates"):
                try:
                    # sync fallback (may expect (subscribe, acctCode) or (subscribe, acctCode, modelCode))
                    try:
                        self.ib.reqAccountUpdates(True, "")
                        traces.append("Called sync reqAccountUpdates(True, '')")
                    except TypeError:
                        try:
                            self.ib.reqAccountUpdates(True, "", "")
                            traces.append("Called sync reqAccountUpdates(True, '', '')")
                        except Exception as e:
                            traces.append(f"sync reqAccountUpdates error: {e}")
                except Exception as e:
                    traces.append(f"sync fallback final error: {e}")
        except Exception as e:
            traces.append(f"outer error: {e}")

        self.log(f"on_connected reqAccountUpdates attempts: {traces}")

    def on_disconnected(self):
        try:
            self.connection_status_signal.emit(False)
        except Exception:
            pass

    # FIXED: Added the missing method that was causing the AttributeError
    def update_connection_status(self, is_connected: bool):
        """Updates UI elements based on IB connection status."""
        try:
            self.connect_btn.setEnabled(not is_connected)
            self.disconnect_btn.setEnabled(is_connected)
            if is_connected:
                self.statusBar().showMessage("Connected to IB.")
            else:
                self.statusBar().showMessage("Disconnected.")
        except Exception as e:
            self.log(f"Error updating connection status UI: {e}")

    def on_position_event(self, pos):
        try:
            sym = getattr(getattr(pos, "contract", None), "symbol", None)
            qty = getattr(pos, "position", 0)
            avg = getattr(pos, "avgCost", 0.0)
            if sym:
                self.position_update_signal.emit(sym, float(qty), float(avg))
        except Exception as e:
            self.log(f"on_position_event error: {e}")

    def on_account_value_event(self, val):
        try:
            if val.tag in ["NetLiquidation", "TotalCashValue", "UnrealizedPnL", "RealizedPnL", "BuyingPower"]:
                self.account_update_signal.emit(val.tag, val.value, val.currency)
        except Exception as e:
            self.log(f"on_account_value_event error: {e}")

    def on_ib_error(self, reqId, errorCode, errorString, contract=None):
        try:
            if 2100 <= errorCode <= 2110 or errorCode == 2158:
                self.log(f"IB Info: {errorString}")
                return
            msg = f"IB Error {errorCode} (Req {reqId}): {errorString}"
            if contract:
                msg += f" | Contract: {getattr(contract, 'symbol', '')}"
            self.log(msg)
        except Exception as e:
            self.log(f"on_ib_error handler error: {e}")

    def on_order_status(self, trade):
        try:
            self.log(f"Order status: {getattr(trade, 'status', '')} id={getattr(trade, 'orderId', '')}")
        except Exception:
            pass

    def on_exec_details(self, det):
        try:
            self.log(f"Exec details: {det}")
        except Exception:
            pass

    # Attempt to bind live bar/historical events (best-effort)
    def _bind_live_bar_events(self):
        if not self.ib:
            return
        # try several possible event names the wrapper might expose
        event_names = ["historicalDataEvent", "historicalData", "realtimeBarEvent", "barUpdateEvent", "realtimeBarsEvent"]
        for name in event_names:
            try:
                evt = getattr(self.ib, name, None)
                if evt is not None:
                    # bind a handler that logs candlestick receptions
                    evt += self._on_live_bar_event
                    self.log(f"Bound live bar handler to event: {name}")
                    break
            except Exception:
                pass

    def _on_live_bar_event(self, *args, **kwargs):
        # The event callback signature varies by wrapper; defensively extract symbol/timeframe if available
        try:
            sym = None
            timeframe = "unknown"
            bar_info = None
            # common shapes: (reqId, bar) or (contract, bar) or (symbol, bar)
            if args:
                for a in args:
                    if hasattr(a, "symbol"):
                        sym = getattr(a, "symbol")
                    if isinstance(a, str):
                        # sometimes first arg is symbol or reqId
                        if a.upper() in self.instruments:
                            sym = a.upper()
                # last arg may be bar-like
                bar_info = args[-1]
            if kwargs and "bar" in kwargs:
                bar_info = kwargs["bar"]
            # try to extract datetime
            dt = None
            if bar_info is not None:
                dt = getattr(bar_info, "time", None) or getattr(bar_info, "date", None) or bar_info.get("date", None) if isinstance(bar_info, dict) else None
            ts = dt if dt is not None else time.strftime("%Y-%m-%d %H:%M:%S")
            if sym:
                self.log(f"Candlestick data received for {sym} (live event) at {ts}")
            else:
                self.log(f"Candlestick data received (live event) at {ts}")
        except Exception:
            pass

    # ---- UI callbacks / instrument management ----
    def on_add_instrument(self):
        s = self.symbol_entry.text().strip().upper()
        sec = self.sec_type.currentText().strip().upper()
        if not s:
            QMessageBox.warning(self, "Input", "Enter a symbol.")
            return
        # assign default data type from global selector
        dtype = self.global_data_type.currentData()
        self.instruments[s] = {"symbol": s, "secType": sec, "dataType": dtype}
        self.log(f"Instrument added: {s} ({sec}), dataType={self.global_data_type.currentText()}")
        self.instrument_list_update_signal.emit(list(self.instruments.keys()))

    def on_remove_selected_instruments(self):
        try:
            selected_rows = sorted(set(idx.row() for idx in self.inst_table.selectedIndexes()), reverse=True)
            removed = []
            for r in selected_rows:
                item = self.inst_table.item(r, 0)
                if item:
                    sym = item.text()
                    removed.append(sym)
                    if sym in self.instruments:
                        del self.instruments[sym]
                    self.inst_table.removeRow(r)
            self.instrument_list_update_signal.emit(list(self.instruments.keys()))
            if removed:
                self.log(f"Removed instruments: {', '.join(removed)}")
        except Exception as e:
            self.log(f"remove selected instruments error: {e}")

    def on_fetch_history_selected(self):
        try:
            rows = sorted(set(idx.row() for idx in self.inst_table.selectedIndexes()))
            if not rows:
                QMessageBox.information(self, "Select", "Select instrument rows to fetch history.")
                return
            for r in rows:
                item = self.inst_table.item(r, 0)
                if item:
                    sym = item.text()
                    # schedule historical fetch; after fetch we log candlestick confirmation
                    self.add_task(self._fetch_and_store_historical(sym))
        except Exception as e:
            self.log(f"fetch history selected error: {e}")

    async def _fetch_and_store_historical(self, symbol: str, duration="1 D", bar_size="5 mins"):
        """
        Fetch historical bars and store DataFrame in self.data[symbol].
        After storing, log a confirmation with latest bar timestamp.
        """
        try:
            if self.ib is None:
                # fake data fallback for dev/testing
                import numpy as np
                # FIXED: Use modern numpy random generator
                if default_rng is None:
                    self.log("NumPy not fully available for fake data generation.")
                    return None
                rng = default_rng()
                n = 200
                times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='5T')
                df = pd.DataFrame(index=times)
                df['open'] = 100 + (np.cumsum(rng.standard_normal(n)) * 0.05)
                df['high'] = df['open'] + (abs(rng.standard_normal(n)) * 0.2)
                df['low'] = df['open'] - (abs(rng.standard_normal(n)) * 0.2)
                df['close'] = df['open'] + (rng.standard_normal(n) * 0.03)
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'datetime'}, inplace=True)
                self.data[symbol] = df
                normalize_ta_columns(df)
                last_time = df['datetime'].iloc[-1]
                self.log(f"Candlestick data received for {symbol} (fake) last bar at {last_time}")
                return df
            
            # Note: This part requires a live IB connection
            contract = self.instruments.get(symbol)
            if contract is None:
                self.log(f"No contract for {symbol}")
                return None
            
            # Construct a proper contract object for ib_async
            sec_type = contract.get("secType", "STK").upper()
            if sec_type == "STK":
                ib_contract = Stock(symbol, 'SMART', 'USD')
            elif sec_type == "CASH":
                ib_contract = Forex(f"{symbol}USD") # Example, adjust as needed
            elif sec_type == "FUT":
                # Futures require more details like exchange, expiry
                # This is a placeholder, adjust with real values
                ib_contract = Future(symbol, '202512', 'CME')
            else:
                self.log(f"Unsupported secType for historical data fetch: {sec_type}")
                return None

            bars = await self.ib.reqHistoricalDataAsync(
                ib_contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )

            if not bars:
                self.log(f"No bars returned for {symbol}")
                return None

            # convert bars to DataFrame
            df = util.df(bars)
            self.data[symbol] = df
            normalize_ta_columns(df)
            if not df.empty:
                last_time = df['date'].iloc[-1]
                self.log(f"Candlestick data received for {symbol} last bar at {last_time} (duration={duration}, bar_size={bar_size})")
            else:
                 self.log(f"Historical data for {symbol} was empty.")
            return df

        except Exception as e:
            self.log(f"_fetch_and_store_historical error for {symbol}: {e}")
            traceback.print_exc()
            return None

    # Apply global data type for instruments missing explicit dataType
    def on_apply_global_data_type(self):
        try:
            val = self.global_data_type.currentData()
            applied = []
            for sym, info in self.instruments.items():
                if info.get("dataType", None) is None:
                    info["dataType"] = val
                    applied.append(sym)
            self.instrument_list_update_signal.emit(list(self.instruments.keys()))
            self.log(f"Applied global data type to: {', '.join(applied) if applied else '<none>'}")
        except Exception as e:
            self.log(f"on_apply_global_data_type error: {e}")

    # ---- Strategy Manager: select-all behavior ----
    def on_select_all_toggled(self, state):
        checked = state == Qt.Checked
        try:
            for cb in self.strategy_checkboxes.values():
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
        except Exception as e:
            self.log(f"on_select_all_toggled error: {e}")

    def _on_individual_strategy_toggled(self, state):
        # If all individual boxes are checked, set select_all; else unset it
        try:
            all_checked = all(cb.isChecked() for cb in self.strategy_checkboxes.values())
            any_unchecked = any(not cb.isChecked() for cb in self.strategy_checkboxes.values())
            # avoid recursion by blocking signal
            self.select_all_chk.blockSignals(True)
            self.select_all_chk.setChecked(all_checked and not any_unchecked)
            self.select_all_chk.blockSignals(False)
        except Exception as e:
            self.log(f"_on_individual_strategy_toggled error: {e}")

    def on_create_checked_strategies(self):
        try:
            checked = [name for name, cb in self.strategy_checkboxes.items() if cb.isChecked()]
            if not checked:
                QMessageBox.information(self, "No strategies", "Select at least one strategy.")
                return
            params = {
                "qty": float(self.qty_spin.value()),
                "interval_seconds": float(self.interval_spin.value()),
                "auto_trade": bool(self.auto_trade_chk.isChecked())
            }
            apply_to_selected = self.apply_to_selected_chk.isChecked()
            symbol_list: List[str] = []
            if apply_to_selected:
                rows = sorted(set(idx.row() for idx in self.inst_table.selectedIndexes()))
                if not rows:
                    QMessageBox.information(self, "Select", "Select instrument rows or disable 'Apply to selected'.")
                    return
                for r in rows:
                    it = self.inst_table.item(r, 0)
                    if it:
                        symbol_list.append(it.text())
            else:
                sym = self.single_symbol_dropdown.currentText().strip()
                if not sym:
                    QMessageBox.information(self, "Select", "Choose a symbol.")
                    return
                symbol_list = [sym]
            for sym in symbol_list:
                for st in checked:
                    self._create_and_start_strategy(sym, st, params)
        except Exception as e:
            self.log(f"on_create_checked_strategies error: {e}")

    def _create_and_start_strategy(self, symbol: str, strat_type: str, params: Dict[str, Any]):
        try:
            strategy = None
            if strat_type == "WMA_SMMA_MACD":
                strategy = WmaSmmaMacdStrategy(self, symbol, params)
            elif strat_type == "TrendlineBreakout":
                strategy = TrendlineBreakoutStrategy(self, symbol, params)
            elif strat_type == "SupportResistance":
                strategy = SupportResistanceStrategy(self, symbol, params)
            elif strat_type == "FairValueGap":
                strategy = FairValueGapStrategy(self, symbol, params)
            else:
                self.log(f"Unknown strategy type {strat_type}")
                return
            sid = self.next_strategy_id
            self.next_strategy_id += 1
            self.strategies[sid] = strategy

            row = self.running_strats_table.rowCount()
            self.running_strats_table.insertRow(row)
            self.running_strats_table.setItem(row, 0, QTableWidgetItem(str(sid)))
            self.running_strats_table.setItem(row, 1, QTableWidgetItem(symbol))
            self.running_strats_table.setItem(row, 2, QTableWidgetItem(strat_type))
            self.running_strats_table.setItem(row, 3, QTableWidgetItem(str(params.get("qty", 1))))
            self.running_strats_table.setItem(row, 4, QTableWidgetItem(str(params.get("interval_seconds", 5.0))))
            stop_btn = QPushButton("Stop")
            def make_stop(sid_local=sid):
                def _stop():
                    self.stop_strategy(sid_local)
                return _stop
            stop_btn.clicked.connect(make_stop())
            self.running_strats_table.setCellWidget(row, 5, stop_btn)

            task = self.add_task(strategy.run())
            if task:
                self.strategy_tasks[sid] = task
                self.log(f"Strategy {sid} started: {strat_type} on {symbol}")
            else:
                self.log(f"Failed to start strategy {strat_type} on {symbol}")
        except Exception as e:
            self.log(f"_create_and_start_strategy error: {e}")

    def stop_strategy(self, sid: int):
        try:
            strategy = self.strategies.get(sid)
            task = self.strategy_tasks.get(sid)
            if strategy:
                try:
                    strategy.stop()
                except Exception:
                    pass
            if task:
                try:
                    task.cancel()
                except Exception:
                    pass
            # remove row from UI
            for r in range(self.running_strats_table.rowCount()):
                it = self.running_strats_table.item(r, 0)
                if it and it.text() == str(sid):
                    self.running_strats_table.removeRow(r)
                    break
            self.strategies.pop(sid, None)
            self.strategy_tasks.pop(sid, None)
            self.log(f"Strategy {sid} stopped and removed.")
        except Exception as e:
            self.log(f"stop_strategy error: {e}")

    # ---- persistence & UI refresh ----
    def save_config(self):
        try:
            conf = {"instruments": self.instruments}
            # FIXED: Use constant for config file name
            with open(CONFIG_FILE, "w") as f:
                json.dump(conf, f)
            self.log("Config saved.")
        except Exception as e:
            self.log(f"save_config error: {e}")

    def load_config(self):
        try:
            # FIXED: Use constant for config file name
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    conf = json.load(f)
                insts = conf.get("instruments", {})
                for s, info in insts.items():
                    self.instruments[s] = info
                self.instrument_list_update_signal.emit(list(self.instruments.keys()))
                self.log("Config loaded.")
        except Exception as e:
            self.log(f"load_config error: {e}")

    def update_instrument_selectors(self, names: List[str]):
        try:
            self.single_symbol_dropdown.clear()
            self.inst_table.setRowCount(0)
            for s in names:
                self.single_symbol_dropdown.addItem(s)
                info = self.instruments.get(s, {})
                row = self.inst_table.rowCount()
                self.inst_table.insertRow(row)
                self.inst_table.setItem(row, 0, QTableWidgetItem(s))
                self.inst_table.setItem(row, 1, QTableWidgetItem(info.get("secType", "")))
                # DataType combobox cell (inline)
                cb = QComboBox()
                for k in DATA_TYPE_MAP:
                    cb.addItem(k, DATA_TYPE_MAP[k])
                dtype = info.get("dataType", None)
                if dtype is None:
                    # global default index
                    gv = self.global_data_type.currentData()
                    for i in range(cb.count()):
                        if cb.itemData(i) == gv:
                            cb.setCurrentIndex(i)
                            break
                else:
                    for i in range(cb.count()):
                        if cb.itemData(i) == dtype:
                            cb.setCurrentIndex(i)
                            break
                def make_on_change(sym=s, combobox=cb):
                    def on_change(idx):
                        try:
                            val = combobox.itemData(idx)
                            if sym in self.instruments:
                                self.instruments[sym]["dataType"] = val
                                self.log(f"Instrument {sym} dataType set to {combobox.currentText()}")
                        except Exception as e:
                            self.log(f"on_change dataType error: {e}")
                    return on_change
                cb.currentIndexChanged.connect(make_on_change())
                self.inst_table.setCellWidget(row, 2, cb)
        except Exception as e:
            self.log(f"update_instrument_selectors error: {e}")

    def update_summary_text(self, text: str):
        self.summary_text.setText(text)

    # ---- account/positions UI updates ----
    def update_account_table(self, tag, value, currency):
        try:
            for r in range(self.account_table.rowCount()):
                it = self.account_table.item(r, 0)
                if it and it.text() == tag:
                    if not self.account_table.item(r, 1):
                        self.account_table.setItem(r, 1, QTableWidgetItem(str(value)))
                    else:
                        self.account_table.item(r, 1).setText(str(value))
                    if not self.account_table.item(r, 2):
                        self.account_table.setItem(r, 2, QTableWidgetItem(str(currency)))
                    else:
                        self.account_table.item(r, 2).setText(str(currency))
                    return
            r = self.account_table.rowCount()
            self.account_table.insertRow(r)
            self.account_table.setItem(r, 0, QTableWidgetItem(tag))
            self.account_table.setItem(r, 1, QTableWidgetItem(str(value)))
            self.account_table.setItem(r, 2, QTableWidgetItem(str(currency)))
        except Exception as e:
            self.log(f"update_account_table error: {e}")

    def update_positions_table(self, sym, qty, avg):
        try:
            for r in range(self.positions_table.rowCount()):
                it = self.positions_table.item(r, 0)
                if it and it.text() == sym:
                    if qty == 0:
                        self.positions_table.removeRow(r)
                    else:
                        if not self.positions_table.item(r, 1):
                            self.positions_table.setItem(r, 1, QTableWidgetItem(str(qty)))
                        else:
                            self.positions_table.item(r, 1).setText(str(qty))
                        if not self.positions_table.item(r, 2):
                            self.positions_table.setItem(r, 2, QTableWidgetItem(str(avg)))
                        else:
                            self.positions_table.item(r, 2).setText(str(avg))
                    return
            if qty != 0:
                r = self.positions_table.rowCount()
                self.positions_table.insertRow(r)
                self.positions_table.setItem(r, 0, QTableWidgetItem(sym))
                self.positions_table.setItem(r, 1, QTableWidgetItem(str(qty)))
                self.positions_table.setItem(r, 2, QTableWidgetItem(str(avg)))
        except Exception as e:
            self.log(f"update_positions_table error: {e}")

# --- Strategies (same defensive implementations as before) ---
class BaseStrategy:
    def __init__(self, app: TradingApp, symbol: str, params: Optional[Dict[str, Any]] = None):
        self.app = app
        self.symbol = symbol
        self.params = params or {}
        self.auto_trade = bool(self.params.get("auto_trade", False))
        self.interval = float(self.params.get("interval_seconds", 5.0))
        self.min_bars = int(self.params.get("min_bars", 50))
        self.is_running = False
        self.contract = self.app.instruments.get(symbol)

    def log(self, msg: str):
        self.app.log(f"[{self.__class__.__name__}:{self.symbol}] {msg}")

    async def place_order(self, action: str, qty: float, order_type: str = "MKT", limit_price: Optional[float] = None):
        try:
            if self.app.ib is None:
                self.log("IB not available — skipping order.")
                return None
            if self.contract is None:
                self.log("No contract for order.")
                return None
            
            # Create a proper contract object for ib_async
            sec_type = self.contract.get("secType", "STK").upper()
            if sec_type == "STK":
                ib_contract = Stock(self.symbol, 'SMART', 'USD')
            elif sec_type == "CASH":
                ib_contract = Forex(f"{self.symbol}USD")
            else:
                self.log(f"Unsupported secType for placing order: {sec_type}")
                return None

            if order_type.upper() == "MKT":
                order = MarketOrder(action, float(qty))
            else:
                order = LimitOrder(action, float(qty), float(limit_price) if limit_price is not None else 0.0)
            
            self.log(f"Placing order: {action} {qty} {self.symbol}")
            trade = self.app.ib.placeOrder(ib_contract, order)
            self.log(f"Trade object: {trade}")
            return trade

        except Exception as e:
            self.log(f"place_order error: {e}")
            traceback.print_exc()
            return None

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        try:
            maybe = self.app.data.get(self.symbol)
            if maybe is None:
                return None
            if isinstance(maybe, pd.DataFrame):
                return maybe.copy()
            try:
                return pd.DataFrame(maybe)
            except Exception:
                return None
        except Exception as e:
            self.log(f"get_dataframe error: {e}")
            return None

    async def run(self):
        self.is_running = True
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            self.is_running = False
        except Exception as e:
            self.log(f"run error: {e}")
            self.is_running = False

    def stop(self):
        self.is_running = False

class WmaSmmaMacdStrategy(BaseStrategy):
    def __init__(self, app, symbol, params=None):
        super().__init__(app, symbol, params)
        self.wma_fast = int(self.params.get("wma_fast", 50))
        self.wma_slow = int(self.params.get("wma_slow", 100))
        self.macd_fast = int(self.params.get("macd_fast", 5))
        self.macd_slow = int(self.params.get("macd_slow", 10))
        self.macd_signal = int(self.params.get("macd_signal", 10))

    def calculate_indicators(self, df: pd.DataFrame):
        try:
            df.ta.wma(length=self.wma_fast, append=True, col_names=(f'WMA_{self.wma_fast}',))
            df.ta.wma(length=self.wma_slow, append=True, col_names=(f'WMA_{self.wma_slow}',))
        except Exception as e:
            self.log(f"WMA calculation error: {e}")
            pass
        try:
            df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)
        except Exception as e:
            self.log(f"MACD calculation error: {e}")
            pass
        normalize_ta_columns(df)

    def check_conditions(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or len(df) < self.min_bars:
            return None
        try:
            price = df['close'].iloc[-1]
            wf_col = f"WMA_{self.wma_fast}"
            ws_col = f"WMA_{self.wma_slow}"
            if wf_col not in df.columns or ws_col not in df.columns or "MACD_hist" not in df.columns:
                self.log(f"Indicator columns missing: needs {wf_col}, {ws_col}, MACD_hist")
                return None
            
            wv = df[wf_col].iloc[-1]
            sv = df[ws_col].iloc[-1]
            macd_hist = df["MACD_hist"].iloc[-1]
            prev_macd_hist = df["MACD_hist"].iloc[-2]

            if price > wv > sv and macd_hist > 0 and macd_hist > prev_macd_hist:
                return "BUY"
            if price < wv < sv and macd_hist < 0 and macd_hist < prev_macd_hist:
                return "SELL"
        except Exception as e:
            self.log(f"check_conditions error: {e}")
        return None

    async def run(self):
        self.is_running = True
        self.log("WMA/SMMA+MACD strategy started.")
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
                df = self.get_dataframe()
                if df is None or len(df) < self.min_bars:
                    continue
                self.calculate_indicators(df)
                sig = self.check_conditions(df)
                if sig:
                    self.log(f"Signal detected: {sig}")
                    qty = float(self.params.get("qty", 1))
                    if self.auto_trade:
                        await self.place_order(sig, qty)
        except asyncio.CancelledError:
            self.log("WMA/SMMA+MACD task cancelled.")
        except Exception as e:
            self.log(f"run error: {e}")
        finally:
            self.is_running = False
            self.log("WMA/SMMA+MACD strategy stopped.")

class TrendlineBreakoutStrategy(BaseStrategy):
    def __init__(self, app, symbol, params=None):
        super().__init__(app, symbol, params)
        self.lookback = int(self.params.get("lookback", 20))
        self.atr_length = int(self.params.get("atr_length", 14))
        self.atr_mult = float(self.params.get("atr_mult", 1.5))

    def calculate_indicators(self, df: pd.DataFrame):
        try:
            df.ta.atr(length=self.atr_length, append=True)
            normalize_ta_columns(df)
        except Exception as e:
            self.log(f"ATR calculation error: {e}")
            pass

    def check_conditions(self, df: pd.DataFrame):
        if df is None or len(df) < self.lookback + 2:
            return None
        try:
            recent = df.iloc[-(self.lookback + 1):]
            highest = recent['high'].iloc[:-1].max()
            lowest = recent['low'].iloc[:-1].min()
            last_close = recent['close'].iloc[-1]
            atr_col = f"ATR_{self.atr_length}"
            atr = df[atr_col].iloc[-1] if atr_col in df.columns else None
            
            if last_close > highest:
                if atr is None or last_close > highest + self.atr_mult * atr:
                    return "BUY"
            if last_close < lowest:
                if atr is None or last_close < lowest - self.atr_mult * atr:
                    return "SELL"
        except Exception as e:
            self.log(f"check_conditions error: {e}")
        return None

    async def run(self):
        self.is_running = True
        self.log("TrendlineBreakout started.")
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
                df = self.get_dataframe()
                if df is None:
                    continue
                self.calculate_indicators(df)
                sig = self.check_conditions(df)
                if sig:
                    self.log(f"Breakout signal: {sig}")
                    if self.auto_trade:
                        await self.place_order(sig, float(self.params.get("qty", 1)))
        except asyncio.CancelledError:
            self.log("TrendlineBreakout task cancelled.")
        except Exception as e:
            self.log(f"run error: {e}")
        finally:
            self.is_running = False
            self.log("TrendlineBreakout stopped.")

class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, app, symbol, params=None):
        super().__init__(app, symbol, params)
        self.period = int(self.params.get("period", 50))
        self.atr_length = 14

    def calculate_indicators(self, df: pd.DataFrame):
        try:
            df.ta.atr(length=self.atr_length, append=True)
            normalize_ta_columns(df)
        except Exception as e:
            self.log(f"ATR calculation error: {e}")


    def check_conditions(self, df: pd.DataFrame):
        if df is None or len(df) < self.period + 2:
            return None
        try:
            recent = df.tail(self.period)
            support = recent['low'].min()
            resistance = recent['high'].max()
            price = df['close'].iloc[-1]
            open_price = df['open'].iloc[-1]
            
            if abs(price - support) < (support * 0.01) and price > open_price:
                return "BUY"
            if abs(resistance - price) < (resistance * 0.01) and price < open_price:
                return "SELL"
        except Exception as e:
            self.log(f"check_conditions error: {e}")
        return None

    async def run(self):
        self.is_running = True
        self.log("SupportResistance started.")
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
                df = self.get_dataframe()
                if df is None:
                    continue
                self.calculate_indicators(df)
                sig = self.check_conditions(df)
                if sig:
                    self.log(f"Support/Resistance signal: {sig}")
                    if self.auto_trade:
                        await self.place_order(sig, float(self.params.get("qty", 1)))
        except asyncio.CancelledError:
            self.log("SupportResistance task cancelled.")
        except Exception as e:
            self.log(f"run error: {e}")
        finally:
            self.is_running = False
            self.log("SupportResistance stopped.")

class FairValueGapStrategy(BaseStrategy):
    def __init__(self, app, symbol, params=None):
        super().__init__(app, symbol, params)
        self.qty = float(self.params.get("qty", 1))

    def detect_fvg(self, df: pd.DataFrame):
        if df is None or len(df) < 3:
            return None
        try:
            # Candle A is 3rd from last, B is 2nd, C is the latest
            a = df.iloc[-3]
            b = df.iloc[-2]
            c = df.iloc[-1]

            # Bullish FVG: low of candle A is higher than high of candle C
            # We are looking for the gap between A's high and C's low
            if a['high'] < c['low']:
                # Price has now moved into the gap
                if b['close'] > a['high'] and b['close'] < c['low']:
                    return "BUY" # More accurately, a bullish imbalance exists
            
            # Bearish FVG: high of candle A is lower than low of candle C
            # We are looking for the gap between A's low and C's high
            if a['low'] > c['high']:
                # Price has now moved into the gap
                if b['close'] < a['low'] and b['close'] > c['high']:
                    return "SELL" # More accurately, a bearish imbalance exists

        except Exception as e:
            self.log(f"detect_fvg error: {e}")
        return None

    async def run(self):
        self.is_running = True
        self.log("FairValueGap started.")
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
                df = self.get_dataframe()
                if df is None:
                    continue
                sig = self.detect_fvg(df)
                if sig:
                    self.log(f"FVG signal: {sig}")
                    if self.auto_trade:
                        await self.place_order(sig, self.qty)
        except asyncio.CancelledError:
            self.log("FairValueGap task cancelled.")
        except Exception as e:
            self.log(f"run error: {e}")
        finally:
            self.is_running = False
            self.log("FairValueGap stopped.")

# ---- Run the app ----
if __name__ == "__main__":
    try:
        qt_app = QApplication(sys.argv)
        loop = qasync.QEventLoop(qt_app)
        asyncio.set_event_loop(loop)
        main_win = TradingApp()
        main_win.show()
        with loop:
            loop.run_forever()
    except Exception as e:
        print("Fatal error running the app:", e)
        traceback.print_exc()
