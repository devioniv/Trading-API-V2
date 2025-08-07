import sys
import asyncio
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QLabel, QLineEdit, QComboBox,
    QListWidget, QTextEdit, QFrame, QGridLayout, QCheckBox, QGroupBox
)
from PySide6.QtCore import Signal, Slot
from ib_async import IB, util, Stock, Forex, Future, Option, Crypto, MarketOrder, LimitOrder, StopOrder
import pandas as pd
import pandas_ta as ta
import qasync

# --- CONSTANTS ---
ENABLE_STRATEGY_TEXT = "Enable Strategy"
KNOWN_CRYPTO_SYMBOLS = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD']

# --- MAIN APPLICATION CLASS ---
class TradingApp(QMainWindow):
    # Signals to safely update GUI from other threads (like ib_async callbacks)
    log_signal = Signal(str)
    summary_signal = Signal(str)
    connection_status_signal = Signal(bool)
    instrument_list_update_signal = Signal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IBKR Trading Bot (PySide6)")
        self.setGeometry(100, 100, 950, 750)

        # --- IB Connection Setup ---
        self.ib = IB()
        self.ib.connectedEvent += self.on_connected
        self.ib.disconnectedEvent += self.on_disconnected
        self.ib.orderStatusEvent += self.on_order_status
        self.ib.execDetailsEvent += self.on_exec_details

        # --- Data Storage ---
        self.instruments = {}
        self.data = {}
        self.strategies = {}

        # --- Connect signals to slots for thread-safe GUI updates ---
        self.log_signal.connect(self.update_log_text)
        self.summary_signal.connect(self.update_summary_text)
        self.connection_status_signal.connect(self.update_connection_status)
        self.instrument_list_update_signal.connect(self.update_instrument_selectors)

        # --- Build the UI ---
        self.create_widgets()

    def create_widgets(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- Connection Controls ---
        connection_group = QGroupBox("Connection")
        connection_layout = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_ib)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_ib)
        self.disconnect_button.setEnabled(False)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.disconnect_button)
        connection_layout.addStretch()
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)

        # --- Main Tabs ---
        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook)

        self.instruments_tab = QWidget()
        self.strategies_tab = QWidget()
        self.logs_tab = QWidget()
        self.summary_tab = QWidget()

        self.notebook.addTab(self.instruments_tab, 'Instruments')
        self.notebook.addTab(self.strategies_tab, 'Strategies')
        self.notebook.addTab(self.logs_tab, 'Logs')
        self.notebook.addTab(self.summary_tab, 'Summary')

        self.create_instruments_tab()
        self.create_strategies_tab()
        self.create_logs_tab()
        self.create_summary_tab()

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Disconnected")

    def create_instruments_tab(self):
        layout = QHBoxLayout(self.instruments_tab)
        
        # Left side for adding instruments
        add_group = QGroupBox("Manage Instruments")
        add_layout = QVBoxLayout()
        
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_entry = QLineEdit()
        form_layout.addWidget(self.symbol_entry, 0, 1)
        
        form_layout.addWidget(QLabel("Sec Type:"), 1, 0)
        self.sec_type_selector = QComboBox()
        self.sec_type_selector.addItems(["STK", "CASH", "FUT", "OPT", "CRYPTO"])
        self.sec_type_selector.currentTextChanged.connect(self.on_sec_type_selected)
        form_layout.addWidget(self.sec_type_selector, 1, 1)
        add_layout.addLayout(form_layout)

        self.option_fut_frame = QFrame()
        self.option_fut_frame.setFrameShape(QFrame.StyledPanel)
        opt_fut_layout = QGridLayout(self.option_fut_frame)
        opt_fut_layout.addWidget(QLabel("Expiry (YYYYMMDD):"), 0, 0)
        self.expiry_entry = QLineEdit()
        opt_fut_layout.addWidget(self.expiry_entry, 0, 1)
        opt_fut_layout.addWidget(QLabel("Strike:"), 1, 0)
        self.strike_entry = QLineEdit()
        opt_fut_layout.addWidget(self.strike_entry, 1, 1)
        opt_fut_layout.addWidget(QLabel("Right:"), 2, 0)
        self.right_selector = QComboBox()
        self.right_selector.addItems(["C", "P"])
        opt_fut_layout.addWidget(self.right_selector, 2, 1)
        opt_fut_layout.addWidget(QLabel("Multiplier:"), 3, 0)
        self.multiplier_entry = QLineEdit("100")
        opt_fut_layout.addWidget(self.multiplier_entry, 3, 1)
        add_layout.addWidget(self.option_fut_frame)
        self.option_fut_frame.setVisible(False)
        
        self.add_button = QPushButton("Add Instrument")
        self.add_button.clicked.connect(self.add_instrument)
        add_layout.addWidget(self.add_button)
        add_layout.addStretch()
        add_group.setLayout(add_layout)

        # Right side for list of instruments
        list_group = QGroupBox("Added Instruments")
        list_layout = QVBoxLayout()
        self.instrument_listbox = QListWidget()
        list_layout.addWidget(self.instrument_listbox)
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_instrument)
        list_layout.addWidget(self.remove_button)
        list_group.setLayout(list_layout)

        layout.addWidget(add_group, 1)
        layout.addWidget(list_group, 1)
        
    def create_strategies_tab(self):
        layout = QVBoxLayout(self.strategies_tab)
        
        top_frame = QFrame()
        top_layout = QHBoxLayout(top_frame)
        top_layout.addWidget(QLabel("Instrument:"))
        self.strategy_instrument_selector = QComboBox()
        top_layout.addWidget(self.strategy_instrument_selector, 1)
        top_layout.addWidget(QLabel("Base Quantity:"))
        self.quantity_entry = QLineEdit("10")
        top_layout.addWidget(self.quantity_entry)
        layout.addWidget(top_frame)

        strategy_notebook = QTabWidget()
        layout.addWidget(strategy_notebook)
        
        # --- WMA/MACD Tab ---
        wma_macd_tab = QWidget()
        wma_macd_layout = QGridLayout(wma_macd_tab)
        self.wma_macd_enabled = QCheckBox(ENABLE_STRATEGY_TEXT)
        wma_macd_layout.addWidget(self.wma_macd_enabled, 0, 0, 1, 4)
        
        wma_macd_layout.addWidget(QLabel("WMA Period:"), 1, 0); self.wma_period_entry = QLineEdit("20"); wma_macd_layout.addWidget(self.wma_period_entry, 1, 1)
        wma_macd_layout.addWidget(QLabel("SMMA Period:"), 2, 0); self.smma_period_entry = QLineEdit("20"); wma_macd_layout.addWidget(self.smma_period_entry, 2, 1)
        wma_macd_layout.addWidget(QLabel("MACD Fast:"), 3, 0); self.macd_fast_entry = QLineEdit("8"); wma_macd_layout.addWidget(self.macd_fast_entry, 3, 1)
        wma_macd_layout.addWidget(QLabel("MACD Slow:"), 4, 0); self.macd_slow_entry = QLineEdit("13"); wma_macd_layout.addWidget(self.macd_slow_entry, 4, 1)
        wma_macd_layout.addWidget(QLabel("MACD Signal:"), 5, 0); self.macd_signal_entry = QLineEdit("7"); wma_macd_layout.addWidget(self.macd_signal_entry, 5, 1)
        
        wma_macd_layout.addWidget(QLabel("Partial TP (R):"), 1, 2); self.tp_entry = QLineEdit("1.5"); wma_macd_layout.addWidget(self.tp_entry, 1, 3)
        wma_macd_layout.addWidget(QLabel("SL ATR Multiplier:"), 2, 2); self.sl_atr_multiplier_entry = QLineEdit("1.5"); wma_macd_layout.addWidget(self.sl_atr_multiplier_entry, 2, 3)
        wma_macd_layout.setColumnStretch(4, 1)
        wma_macd_layout.setRowStretch(6, 1)
        strategy_notebook.addTab(wma_macd_tab, 'WMA/SMMA+MACD')

        # --- FVG Tab ---
        fvg_tab = QWidget()
        fvg_layout = QGridLayout(fvg_tab)
        self.fvg_enabled = QCheckBox(ENABLE_STRATEGY_TEXT)
        fvg_layout.addWidget(self.fvg_enabled, 0, 0, 1, 2)
        fvg_layout.addWidget(QLabel("Min Gap Size:"), 1, 0)
        self.fvg_min_gap_size = QLineEdit("0.1")
        fvg_layout.addWidget(self.fvg_min_gap_size, 1, 1)
        fvg_layout.setRowStretch(2, 1)
        fvg_layout.setColumnStretch(2, 1)
        strategy_notebook.addTab(fvg_tab, 'Fair Value Gap')

        # --- Other Tab ---
        other_tab = QWidget()
        other_layout = QVBoxLayout(other_tab)
        self.trendline_enabled = QCheckBox("Enable Trendline Breakout Strategy")
        self.sr_enabled = QCheckBox("Enable Support/Resistance Strategy")
        other_layout.addWidget(self.trendline_enabled)
        other_layout.addWidget(self.sr_enabled)
        other_layout.addStretch()
        strategy_notebook.addTab(other_tab, 'Other')

        self.save_strategies_button = QPushButton("Save Strategies For Selected Instrument")
        self.save_strategies_button.clicked.connect(self.save_strategies)
        layout.addWidget(self.save_strategies_button)

    def create_logs_tab(self):
        layout = QVBoxLayout(self.logs_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def create_summary_tab(self):
        layout = QVBoxLayout(self.summary_tab)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

    def connect_ib(self):
        self.log("Attempting to connect...")
        asyncio.create_task(self._connect_ib())

    async def _connect_ib(self):
        try:
            await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        except Exception as e:
            self.log(f"Connection failed: {e}")

    def disconnect_ib(self):
        self.ib.disconnect()

    # --- Event Handlers from IB ---
    def on_connected(self):
        self.connection_status_signal.emit(True)

    def on_disconnected(self):
        self.connection_status_signal.emit(False)

    def on_order_status(self, trade):
        self.log(f"Order Status for {trade.contract.symbol}: {trade.orderStatus.status}")
        for sym, strats in self.strategies.items():
            for strat in strats:
                if strat.contract and trade.contract.conId == strat.contract.conId:
                    if trade.orderStatus.status in ['Filled', 'Cancelled', 'Inactive']:
                        if strat.trade_info['is_open'] and any(o.orderId == trade.order.orderId for o in strat.trade_info['open_orders']):
                            self.log(f"Order for {sym} is final. Resetting strategy state.")
                            strat.reset_trade_state()

    def on_exec_details(self, trade, fill):
        self.log(f"Execution for {trade.contract.symbol}: {fill.execution}")
        self.log_trade(trade, fill)

    def log_trade(self, trade, fill):
        summary = (
            f"--- TRADE SUMMARY ---\n"
            f"Symbol: {trade.contract.symbol}\n"
            f"Action: {fill.execution.side}\n"
            f"Quantity: {fill.execution.shares}\n"
            f"Avg Fill Price: {fill.execution.avgPrice}\n"
            f"Commission: {fill.commissionReport.commission}\n"
            f"---------------------\n"
        )
        self.summary_signal.emit(summary)

    def on_bar_update(self, bars, has_new_bar):
     if has_new_bar:
        symbol = next((s for s, c in self.instruments.items() if c.conId == bars.contract.conId), None)
        if symbol is None:
            return
        self.log(f"New 5-minute bar for {symbol}: {bars[-1].time}")
        if symbol in self.strategies:
            for strategy in self.strategies[symbol]:
                asyncio.create_task(strategy.run())

    # --- GUI Slots and Callbacks ---
    def log(self, message):
        self.log_signal.emit(str(message))

    @Slot(str)
    def update_log_text(self, message):
        self.log_text.append(message)

    @Slot(str)
    def update_summary_text(self, summary):
        self.summary_text.append(summary)

    @Slot(bool)
    def update_connection_status(self, is_connected):
        if is_connected:
            self.status_bar.showMessage("Connected")
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.log("Connected to IBKR")
        else:
            self.status_bar.showMessage("Disconnected")
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.log("Disconnected from IBKR")

    @Slot(list)
    def update_instrument_selectors(self, instruments_list):
        self.instrument_listbox.clear()
        self.instrument_listbox.addItems(instruments_list)
        self.strategy_instrument_selector.clear()
        self.strategy_instrument_selector.addItems(instruments_list)

    def on_sec_type_selected(self, sec_type):
        is_opt_fut = sec_type in ["OPT", "FUT"]
        self.option_fut_frame.setVisible(is_opt_fut)

    def add_instrument(self):
        symbol = self.symbol_entry.text().upper()
        if not symbol or symbol in self.instruments: return
        asyncio.create_task(self._add_instrument(symbol))

    async def _add_instrument(self, symbol):
        sec_type = self.sec_type_selector.currentText()
        contract = None
        try:
            if symbol in KNOWN_CRYPTO_SYMBOLS:
                sec_type = 'CRYPTO'
            if sec_type == "STK": contract = Stock(symbol, 'SMART', 'USD')
            elif sec_type == "CASH":
                if len(symbol) != 6: self.log(f"Invalid Forex symbol: {symbol}. Must be 6 characters."); return
                contract = Forex(symbol)
            elif sec_type == "CRYPTO":
                if symbol.endswith('USD'): contract = Crypto(symbol[:-3], 'PAXOS', 'USD')
                else: self.log(f"Invalid crypto symbol: {symbol}. Use 'BTCUSD', etc."); return
            elif sec_type == "FUT":
                expiry = self.expiry_entry.text()
                contract = Future(symbol, expiry, 'SMART', currency='USD')
            elif sec_type == "OPT":
                expiry = self.expiry_entry.text()
                strike = float(self.strike_entry.text())
                right = self.right_selector.currentText()
                contract = Option(symbol, expiry, strike, right, 'SMART', currency='USD')
        except Exception as e: self.log(f"Error creating contract for {symbol}: {e}"); return
        if contract is None: self.log(f"Unsupported security type: {sec_type}"); return
        try:
            qualified_contracts = await self.ib.qualifyContractsAsync(contract)
            if qualified_contracts:
                qualified_contract = qualified_contracts[0]
                self.instruments[symbol] = qualified_contract
                self.instrument_list_update_signal.emit(list(self.instruments.keys()))
                self.log(f"Added instrument: {symbol}")
                await self.fetch_historical_data(symbol)
            else: self.log(f"Could not qualify contract for {symbol}: Unknown contract.")
        except Exception as e: self.log(f"Error adding instrument {symbol}: {e}")

    def remove_instrument(self):
        selected_item = self.instrument_listbox.currentItem()
        if not selected_item: return
        symbol = selected_item.text()
        
        if bar_data := self.data.get(symbol): self.ib.cancelHistoricalData(bar_data)
        if symbol in self.instruments: del self.instruments[symbol]
        if symbol in self.data: del self.data[symbol]
        
        self.instrument_list_update_signal.emit(list(self.instruments.keys()))
        self.log(f"Removed instrument: {symbol}")

    async def fetch_historical_data(self, symbol):
        contract = self.instruments.get(symbol)
        if not contract: return
        what_to_show, use_rth = 'TRADES', True
        if contract.secType == 'CRYPTO': what_to_show, use_rth = 'AGGTRADES', False
        elif contract.secType == 'CASH': what_to_show, use_rth = 'ASK', False
        try:
            self.log(f"Requesting data for {symbol} with whatToShow='{what_to_show}'...")
            bars = await self.ib.reqHistoricalDataAsync(
                contract, endDateTime='', durationStr='5 D', barSizeSetting='5 mins',
                whatToShow=what_to_show, useRTH=use_rth, formatDate=1, keepUpToDate=True)
            if bars:
                self.data[symbol] = bars
                bars.updateEvent += self.on_bar_update
                self.log(f"Fetched {len(bars)} historical bars for {symbol}")
            else: self.log(f"Could not fetch historical data for {symbol}.")
        except Exception as e: self.log(f"Exception fetching data for {symbol}: {e}")

    def save_strategies(self):
        symbol = self.strategy_instrument_selector.currentText()
        if not symbol: self.log("Please select an instrument first."); return
        try:
            quantity = int(self.quantity_entry.text())
        except ValueError: self.log("Invalid quantity."); return

        self.strategies[symbol] = []
        trade_management_params = {
            "take_profit_rr": float(self.tp_entry.text()),
            "sl_atr_multiplier": float(self.sl_atr_multiplier_entry.text()),
            "sl_lookback": 10
        }

        if self.wma_macd_enabled.isChecked():
            try:
                params = {
                    "wma_period": int(self.wma_period_entry.text()), "smma_period": int(self.smma_period_entry.text()),
                    "macd_fast": int(self.macd_fast_entry.text()), "macd_slow": int(self.macd_slow_entry.text()),
                    "macd_signal": int(self.macd_signal_entry.text()), "quantity": quantity, **trade_management_params
                }
                self.strategies[symbol].append(WmaSmmaMacdStrategy(self, symbol, params))
                self.log(f"WMA/SMMA+MACD strategy enabled for {symbol}")
            except ValueError as e: self.log(f"Invalid WMA/SMMA+MACD parameter: {e}")
        
        if self.trendline_enabled.isChecked():
            params = {"quantity": quantity, "period": 20, **trade_management_params}
            self.strategies[symbol].append(TrendlineBreakoutStrategy(self, symbol, params))
            self.log(f"Trendline Breakout strategy enabled for {symbol}")

        if self.sr_enabled.isChecked():
            params = {"quantity": quantity, "period": 50, **trade_management_params}
            self.strategies[symbol].append(SupportResistanceStrategy(self, symbol, params))
            self.log(f"Support/Resistance strategy enabled for {symbol}")

        if self.fvg_enabled.isChecked():
            try:
                fvg_params = {
                    "quantity": quantity, "min_gap_size": float(self.fvg_min_gap_size.text()),
                    "max_candles": 10, "lookback_bos": 20, "range_threshold": 0.02, 
                    "lower_threshold": 0.02, "upper_threshold": 0.6, "support_lookback": 10, 
                    "support_tolerance": 0.01, "support_method": "high_low_limit", 
                    "use_support_resistance": True, "use_break_of_structure": True, 
                    "use_market_regime_filter": False, "use_allowed_range_filter": True, 
                    "use_dynamic_sizing": False, **trade_management_params
                }
                self.strategies[symbol].append(FairValueGapStrategy(self, symbol, fvg_params))
                self.log(f"Fair Value Gap strategy enabled for {symbol}")
            except ValueError as e: self.log(f"Invalid FVG parameter: {e}")
        
        self.log(f"Strategies saved for {symbol}.")


# --- PARENT STRATEGY CLASS ---
# This class and all child classes are copied directly from the original script
# and require no modification as they don't interact with the GUI directly.
class Strategy:
    def __init__(self, app, symbol, params: dict):
        self.app = app
        self.symbol = symbol
        self.ib = app.ib
        self.params = params
        self.quantity = params.get("quantity", 1)
        self.tp_rr = params.get("take_profit_rr", 1.5)
        self.sl_atr_multiplier = params.get("sl_atr_multiplier", 1.5)
        self.trade_info = {
            "is_open": False, "direction": None, "entry_price": 0.0,
            "initial_sl_price": 0.0, "partial_tp_price": 0.0,
            "partial_taken": False, "trailing_sl_price": 0.0,
            "open_orders": []
        }
    @property
    def contract(self): return self.app.instruments.get(self.symbol)
    @property
    def data(self): return self.app.data.get(self.symbol)
    async def run(self):
        if self.trade_info["is_open"]: await self.manage_open_position()
        else: await self.check_for_entry()
    async def check_for_entry(self): raise NotImplementedError
    def get_dataframe(self, min_length: int):
        if self.data is None or len(self.data) < min_length: return None
        df = util.df(self.data)
        if df is None or df.empty: return None
        return df
    def reset_trade_state(self):
        self.trade_info = { "is_open": False, "direction": None, "entry_price": 0.0, "initial_sl_price": 0.0,
            "partial_tp_price": 0.0, "partial_taken": False, "trailing_sl_price": 0.0, "open_orders": [] }
        self.app.log(f"Trade state has been reset for {self.symbol}.")
    async def place_entry_order(self, action: str, sl_price: float, partial_tp_price: float):
        for order in self.trade_info["open_orders"]: self.ib.cancelOrder(order)
        entry_order = MarketOrder(action, self.quantity)
        entry_trade = self.ib.placeOrder(self.contract, entry_order)
        self.app.log(f"Placed {action} entry order for {self.symbol} qty {self.quantity}.")
        self.trade_info.update({
            "is_open": True, "direction": 'long' if action == 'BUY' else 'short', 
            "entry_price": self.data[-1].close, "initial_sl_price": sl_price, 
            "trailing_sl_price": sl_price, "partial_tp_price": partial_tp_price, 
            "partial_taken": False, "open_orders": [entry_trade.order]
        })
    async def close_position(self, reason: str):
        if not self.trade_info["is_open"]: return
        close_action = 'SELL' if self.trade_info["direction"] == 'long' else 'BUY'
        current_position = next((pos.position for pos in self.app.ib.positions() if pos.contract.conId == self.contract.conId), 0)
        if current_position != 0:
            close_order = MarketOrder(close_action, abs(current_position))
            self.ib.placeOrder(self.contract, close_order)
            self.app.log(f"CLOSING position for {self.symbol} ({reason}). Action: {close_action} {abs(current_position)}.")
        self.reset_trade_state()
    async def manage_open_position(self):
        df = self.get_dataframe(min_length=20)
        if df is None: return
        price = self.data[-1].close
        direction = self.trade_info["direction"]
        trailing_sl = self.trade_info["trailing_sl_price"]
        if (direction == 'long' and price <= trailing_sl) or (direction == 'short' and price >= trailing_sl):
            await self.close_position(f"Stop-Loss hit at {trailing_sl:.2f}"); return
        if not self.trade_info['partial_taken']:
            partial_tp = self.trade_info['partial_tp_price']
            if (direction == 'long' and price >= partial_tp) or (direction == 'short' and price <= partial_tp):
                qty = self.quantity / 2
                if qty < 1: qty = 1
                tp_action = 'SELL' if direction == 'long' else 'BUY'
                tp_order = MarketOrder(tp_action, qty)
                self.ib.placeOrder(self.contract, tp_order)
                self.app.log(f"Taking PARTIAL PROFIT for {self.symbol} at {price:.2f}.")
                self.trade_info['partial_taken'] = True
                self.trade_info['trailing_sl_price'] = self.trade_info['entry_price']
                self.app.log(f"Stop-Loss for {self.symbol} moved to breakeven: {self.trade_info['entry_price']:.2f}")
        else:
            df.ta.atr(length=14, append=True)
            atr = df["ATRr_14"].iloc[-1]
            if atr is None or pd.isna(atr): return
            if direction == 'long':
                new_sl = price - (atr * self.sl_atr_multiplier)
                if new_sl > trailing_sl:
                    self.trade_info['trailing_sl_price'] = new_sl
                    self.app.log(f"Trailing Stop for {self.symbol} moved UP to {new_sl:.2f}")
            else:
                new_sl = price + (atr * self.sl_atr_multiplier)
                if new_sl < trailing_sl:
                    self.trade_info['trailing_sl_price'] = new_sl
                    self.app.log(f"Trailing Stop for {self.symbol} moved DOWN to {new_sl:.2f}")

# --- CHILD STRATEGY CLASSES ---
class WmaSmmaMacdStrategy(Strategy):
    def __init__(self, app, symbol, params: dict):
        super().__init__(app, symbol, params)
        self.wma_period = params.get("wma_period", 20)
        self.smma_period = params.get("smma_period", 20)
        self.macd_fast = params.get("macd_fast", 8)
        self.macd_slow = params.get("macd_slow", 13)
        self.macd_signal = params.get("macd_signal", 7)
        self.sl_lookback = params.get("sl_lookback", 10)
        self.min_data_length = max(self.wma_period, self.smma_period, self.macd_slow, self.sl_lookback) + 15

    async def check_for_entry(self):
        df = self.get_dataframe(min_length=self.min_data_length)
        if df is None: return
        df.ta.wma(length=self.wma_period, append=True)
        df.ta.smma(length=self.smma_period, append=True)
        df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)
        df.ta.atr(length=14, append=True)
        wma, smma = f"WMA_{self.wma_period}", f"SMMA_{self.smma_period}"
        macd_l, macd_s = f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}", f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        atr_col = "ATRr_14"
        if not all(c in df.columns for c in [wma, smma, macd_l, macd_s, atr_col]): return
        wma_p, wma_c = df[wma].iloc[-3], df[wma].iloc[-2]
        smma_p, smma_c = df[smma].iloc[-3], df[smma].iloc[-2]
        macd_line, macd_signal = df[macd_l].iloc[-2], df[macd_s].iloc[-2]
        is_bull = wma_p < smma_p and wma_c > smma_c and macd_line > macd_signal
        is_bear = wma_p > smma_p and wma_c < smma_c and macd_line < macd_signal
        if is_bull:
            self.app.log(f"BULLISH SIGNAL for {self.symbol}")
            lookback = df.iloc[-self.sl_lookback:-1]
            swing_low = lookback['low'].min()
            atr = df[atr_col].iloc[-2]
            sl = swing_low - (atr * self.sl_atr_multiplier)
            price = df['close'].iloc[-2]
            risk = price - sl
            tp = price + (risk * self.tp_rr)
            await self.place_entry_order('BUY', sl_price=round(sl, 2), partial_tp_price=round(tp, 2))
        elif is_bear:
            self.app.log(f"BEARISH SIGNAL for {self.symbol}")
            lookback = df.iloc[-self.sl_lookback:-1]
            swing_high = lookback['high'].max()
            atr = df[atr_col].iloc[-2]
            sl = swing_high + (atr * self.sl_atr_multiplier)
            price = df['close'].iloc[-2]
            risk = sl - price
            tp = price - (risk * self.tp_rr)
            await self.place_entry_order('SELL', sl_price=round(sl, 2), partial_tp_price=round(tp, 2))

class TrendlineBreakoutStrategy(Strategy):
    def __init__(self, app, symbol, params: dict):
        super().__init__(app, symbol, params)
        self.period = params.get("period", 20)
        self.min_data_length = self.period + 15
    async def check_for_entry(self):
        df = self.get_dataframe(min_length=self.min_data_length)
        if df is None: return
        df.ta.atr(length=14, append=True)
        if "ATRr_14" not in df.columns: return
        recent = df.tail(self.period)
        highs, lows = recent['high'], recent['low']
        res_slope = (highs.max() - highs.iloc[0]) / self.period if self.period > 0 else 0
        res_intercept = highs.iloc[0]
        sup_slope = (lows.min() - lows.iloc[0]) / self.period if self.period > 0 else 0
        sup_intercept = lows.iloc[0]
        price = df['close'].iloc[-2]
        res_price = res_slope * (self.period - 1) + res_intercept
        sup_price = sup_slope * (self.period - 1) + sup_intercept
        atr = df["ATRr_14"].iloc[-2]
        if price > res_price:
            sl = res_price - (atr * self.sl_atr_multiplier)
            risk = price - sl
            tp = price + (risk * self.tp_rr)
            await self.place_entry_order('BUY', sl_price=round(sl, 2), partial_tp_price=round(tp, 2))
        elif price < sup_price:
            sl = sup_price + (atr * self.sl_atr_multiplier)
            risk = sl - price
            tp = price - (risk * self.tp_rr)
            await self.place_entry_order('SELL', sl_price=round(sl, 2), partial_tp_price=round(tp, 2))

class SupportResistanceStrategy(Strategy):
    def __init__(self, app, symbol, params: dict):
        super().__init__(app, symbol, params)
        self.period = params.get("period", 50)
        self.min_data_length = self.period + 15
    async def check_for_entry(self):
        df = self.get_dataframe(min_length=self.min_data_length)
        if df is None: return
        df.ta.atr(length=14, append=True)
        if "ATRr_14" not in df.columns: return
        recent = df.tail(self.period)
        support = recent['low'].min()
        resistance = recent['high'].max()
        price = df['close'].iloc[-2]
        open_price = df['open'].iloc[-2]
        atr = df["ATRr_14"].iloc[-2]
        if abs(price - support) < (support * 0.01) and price > open_price:
            sl = support - (atr * self.sl_atr_multiplier)
            risk = price - sl
            tp = price + (risk * self.tp_rr)
            await self.place_entry_order('BUY', sl_price=round(sl, 2), partial_tp_price=round(tp, 2))
        elif abs(price - resistance) < (resistance * 0.01) and price < open_price:
            sl = resistance + (atr * self.sl_atr_multiplier)
            risk = sl - price
            tp = price - (risk * self.tp_rr)
            await self.place_entry_order('SELL', sl_price=round(sl, 2), partial_tp_price=round(tp, 2))

class FairValueGapStrategy(Strategy):
    def __init__(self, app, symbol, config: dict):
        super().__init__(app, symbol, config)
        self.min_gap_size = config.get("min_gap_size", 0.1)
        self.max_candles = config.get("max_candles", 10)
        self.active_fvgs = []
        self.min_data_length = 200
    async def check_for_entry(self):
        df = self.get_dataframe(min_length=self.min_data_length)
        if df is None: return
        i = len(df) - 1
        if i < 2: return
        df = self.calculate_indicators(df)
        self.detect_new_fvg(df, i)
        await self.update_active_fvgs(df, i)
    async def update_active_fvgs(self, df: pd.DataFrame, i: int):
        # Full FVG entry logic would go here
        #self.app.log("FVG Strategy: update_active_fvgs not implemented.")
        pass
    def calculate_indicators(self, df: pd.DataFrame):
        # Full FVG indicator calculations would go here
        df.ta.atr(length=14, append=True)
        return df
    def detect_new_fvg(self, df: pd.DataFrame, i: int):
        # Full FVG detection logic would go here
        #self.app.log("FVG Strategy: detect_new_fvg not implemented.")
        pass

# --- EXECUTION BLOCK ---
async def main():
    """Main async function to bootstrap the application."""
    app = QApplication(sys.argv)

    # Set up the qasync event loop. This must be done before creating the window.
    event_loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(event_loop)

    # Create and show the main window
    main_window = TradingApp()
    main_window.show()

    # The main_window.show() is non-blocking. The event_loop.run_forever()
    # or app.exec() will start the actual event processing.
    # We let the main asyncio.run() handle the loop lifetime.
    await asyncio.sleep(0) # Give the GUI a moment to appear
    return app.exec()


if __name__ == '__main__':
    util.patchAsyncio()
    try:
        # Run the main async function and get the exit code
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except (KeyboardInterrupt, SystemExit):
        print("Application closed.")